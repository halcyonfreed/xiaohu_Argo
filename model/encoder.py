from typing import Optional
from torch_geometric.typing import Adj,Size,OptTensor 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
import torch

from utils import init_weights
from model.embedding import SingleInputEmbedding,MultipleInputEmbedding

class AAEncoder(MessagePassing):
    '''
    encoder的part 1
    论文第三章3.2的 aa: agent-agent interaction 用Pyg的MessagePassing在图的节点之间传递info 是一种GCN还是GAT???
    '''
    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder,self).__init__(aggr='add', node_dim=0,**kwargs) #**kwargs以dict形式传参
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)

        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)  # 这是什么?
        self.attn_drop = nn.Dropout(dropout)
        
        # 用在融合相邻车mit和自车zit时的gate的参数
        self.lin_ih = nn.Linear(embed_dim, embed_dim) # ih: input-hidden
        self.lin_hh = nn.Linear(embed_dim, embed_dim) # hh: hidden-hidden

        # 输出层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim) # why用两个
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)) # 最后一层不要再加relu了 会有问题?why

        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim)) # why用Parameter管理这个tensor,用于init
        nn.init.normal_(self.bos_token,mean=0.,std=.02) # 可以改,对bos_token矩阵填满 normal正态分布取的值
        self.apply(init_weights)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        '''
        mha:multi-head attention
        '''
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size)) # propagate是pyg里的
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        '''
        feedfoward network, mlp就是FFN的一种体现
        '''
        return self.mlp(x)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        '''
        求自车central_embed
        '''
        # ?why 没懂parallel 同时处理多帧t
        if self.parallel:
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))  # ?why .view改形状
            else:
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2)) # ?why *rotate_mat.shape debug了以后还是没懂
            
            
            # 算自车的位置 原view但是报错 提时让用reshape  ?why没懂这里！！
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),self.bos_token.unsqueeze(-2),center_embed).reshape(x.shape[0], -1) # .view(x.shape[0], -1)
                                       
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)

        # 补充Interaction的信息
        center_embed=center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat,size)+\
                self._ff_block(self.norm2(center_embed))
        return center_embed
    
    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        '''
        求nbr_embed-> nbr对自车central的相关性 attention scores α
        '''
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])

        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        # 算α相关性了 用qkv
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        # ?why scale 降采样？？
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i) # 是pyg里特别的稀疏softmax
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        '''
        gate 是Wgate融合nbr周围感兴趣车的交互mit和自车的信息zit
        '''
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

class TemporalEncoderLayer(nn.Module):
    '''
    encoder的part 2
    输出给AL encoder s_i^(T+1)
    '''
    
