from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from embedding import MultipleInputEmbedding
from embedding import SingleInputEmbedding
from utils import TemporalData
from utils import init_weights

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj,Size,OptTensor 
from torch_geometric.utils import softmax, subgraph
# opt是ouput, adj是adjacent

class GraphInteractorLayer(MessagePassing):
    '''
    类似pyg写的 _mha/ff_block自己写的
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int=8,
                 dropout:float=0.1,
                 **kwargs)->None:
        '''改 8个头 dropout防过拟合'''
        # ?why node_dim=0
        super(GraphInteractorLayer,self).__init__(aggr='add',node_dim=0,**kwargs)
        self.embed_dim=embed_dim
        self.num_heads=num_heads

        # ?why lin
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        # why lin_self
        self.lin_self = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        # why
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        # why 叫这个名
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # why *4
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
    
    def _mha_block(self,
                   x:torch.Tensor,
                   edge_index: Adj,
                   edge_attr:torch.Tensor,
                   size:Size=None)->torch.Tensor:
        '''why'''
        # the initial call to start propagating messages.
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))        
        return self.proj_drop(x)
    
    def _ff_block(self,x: torch.Tensor) -> torch.Tensor:
        '''why干嘛的'''
        return self.mlp(x)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr:torch.Tensor,
                size:Size=None)->torch.Tensor:
        # adj是tensor/sparse tensor邻接矩阵；size是int组成的tuple
        # why +self._mha_block
        x=x+self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        x = x + self._ff_block(self.norm2(x))        
        return x

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        # mlp的query key value qkv
        query=self.lin_q_node(x_i).view(-1,self.num_heads, self.embed_dim // self.num_heads) # i自车
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads) # j感兴趣的他车
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)

        # why ?
        scale=(self.embed_dim // self.num_heads) ** 0.5 # why
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1) 
    
    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        # why 怎么算的
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)
    
class GraphInteractor(nn.Module):
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super(GraphInteractor, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes # 这是个什么东西

        # why rotate??
        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        
        self.global_interactor_layers = nn.ModuleList(
            [GraphInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim) # why *num modes
        self.apply(init_weights)
    
    def forward(self,
                data: TemporalData,
                local_embed: torch.Tensor) -> torch.Tensor:
        # why subgraph
        edge_index,_=subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=data.edge_index)
        rel_pos=data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1] # rel是relative相对
        
        if data['rotate_mat'] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_pos=torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
            rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])
        
        x=local_embed
        for layer in self.global_interactor_layers:
            x=layer(x,edge_index,rel_embed)
        x = self.norm(x)  # [N, D] N节点数 D=embed_dim
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N,D]->[N,N*D]->[N, F, D] ?why F是什么
        x=x.transpose(0,1) # [F,N,D]
        return x