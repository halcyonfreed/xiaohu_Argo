from typing import Optional,Tuple
from torch_geometric.typing import Adj,Size,OptTensor 
from torch_geometric.nn.conv import MessagePassing # 不同agent lane之间的interaction用pyg的消息传递
from torch_geometric.data import Data,Batch
from torch_geometric.utils import softmax,subgraph

import torch.nn.functional as F
import torch.nn as nn
import torch

from utils import init_weights,TemporalData,DistanceDropEdge
from models.embedding import SingleInputEmbedding,MultipleInputEmbedding

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
        计算了邻居节点j 他车 到中心节点i自车的 α
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
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads) # [*,8,d_k]
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)  # [*,8,d_k]
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)  # [*,8,d_k]

        scale = (self.embed_dim // self.num_heads) ** 0.5 # 根号d_k
        alpha = (query * key).sum(dim=-1) / scale # 逐元素相乘 维度不变 [*,8,d_k]->[*,8] 因为取了sum
        alpha = softmax(alpha, index, ptr, size_i) # 是pyg里特别的稀疏softmax [*,8]
        alpha = self.attn_drop(alpha) # [*,8]
        return value * alpha.unsqueeze(-1) #alpha[*,8]->[*,8,1]自动广播复制k个对齐，value[*,8,k] return[*,8,k]

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        '''
        gate 是Wgate融合nbr周围感兴趣车的交互mit和自车的信息zit
        '''
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

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

class TemporalEncoderLayer(nn.Module):
    '''
    encoder的part 2
    输出给AL encoder s_i^(T+1)
    '''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        '''
        设置每个layer的参数
        '''
        super(TemporalEncoderLayer,self).__init__()
        nn.Transformer
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
    
    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        '''
        self_attention block, 筛选哪些时刻感兴趣,作为key 求attention scores
        '''
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0] # 
        return self.dropout1(x)
    
    def _ff_block(self,x:torch.Tensor)->torch.Tensor:
        '''
        feed_forward类似MLP
        '''
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)

class TemporalEncoder(nn.Module):
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,norm=nn.LayerNorm(embed_dim))

        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim)) # 都用parameter管理起来
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))

        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)  # 只能看到历史的 看不到未来的  妙！！！
        self.register_buffer('attn_mask', attn_mask)  # attn_mask不更新 but存成model的parameter
        
        nn.init.normal_(self.padding_token, mean=0., std=.02) # 初始化 是正太分布normal distribution
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        '''
        学bert自己加了一个s_i^(T+1) 输出这个s_i^(T+1)
        '''
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1) # 最后加一个
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D] 输出最后一个s_i^(T+1) D??什么意思
        
    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        '''
        求attn_mask 即Huv矩阵的 只看历史的 非历史的要mask不能做attn
        '''
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # 可以连续.masked_fill
        return mask
    
class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        
        # x_i 自己 x_j其他 自己算query　其他算key,value
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads) # [,8,/8]
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads) #[,8,/8]
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads) #[,8,/8]

        scale = (self.embed_dim // self.num_heads) ** 0.5 # 公式根号dk那里 
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)

class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

    def forward(self, data: TemporalData) -> torch.Tensor:
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes)
            batch = Batch.from_data_list(snapshots)
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                         bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
            out = torch.stack(out)  # [T, N, D]
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                              is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                              traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        return out