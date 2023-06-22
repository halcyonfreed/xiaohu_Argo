# Copyright (c) 2023, Siyuan Hu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch_geometric.data import Data
from typing import List,Optional,Tuple # 检查type的error
import torch.nn as nn

class TemporalData(Data):
    # ?why要这个temporal data时序数据！！
    # 根据process_argoverse return来定形参是什么！！ 形参以dict形式
    def __init__(self, 
                 x:Optional[torch.Tensor]=None,
                 positions:Optional[torch.Tensor]=None,
                 edge_index:Optional[torch.Tensor]=None, 
                 edge_attrs:Optional[List[torch.Tensor]]=None, 
                 y:Optional[torch.Tensor]=None, 
                 num_nodes:Optional[int]=None,
                 padding_mask:Optional[torch.Tensor]=None,
                 bos_mask:Optional[torch.Tensor]=None,
                 rotate_angles:Optional[torch.Tensor]=None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        
        if x is None:
            # ?why 为什么x会None
            super(TemporalData,self).__init__()
            return
        super(TemporalData,self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                          padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                          lane_vectors=lane_vectors, is_intersections=is_intersections,
                                          turn_directions=turn_directions, traffic_controls=traffic_controls,
                                          lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                          seq_id=seq_id, **kwargs)
        # GNN的graph的边的属性, 是None啊，要这个干嘛的 ?why
        if edge_attrs is not None:
            # N辆车 N个节点！ size(1)是20 ?why 20
            for t in range(self.x.size(1)): 
                self[f'edge_attr_{t}']=edge_attrs[t]
    
    def __inc__(self,key,value):
        if key=='lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)],[self.num_nodes]]) # ?why 没懂
        else:
            return super().__inc__(key,value)
        
def init_weights(m:nn.Module)->None:
    # 太强了！
    '''
    对model的不同组成模块m(module) 的参数的权重设置不同的初值 改
    '''
    if isinstance(m,nn.Linear):
        # embedding 有
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.LayerNorm):
        # embedding有
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    