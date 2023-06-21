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
# import都是要一个补一个
from torch_geometric.data import Data,Dataset

import os
# typing便于明确type
from typing import Optional,Callable,Union,List,Tuple,Dict

from argoverse.map_representation.map_api import ArgoverseMap
# show progress bar
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from itertools import permutations
from utils import TemporalData

class ArgoverseV1Dataset(Dataset):
    '''
    __init__定义成员 一般是x,y但是这里没写;len;get;process_argoverse自己加的一般没有
    '''
    def __init__(self,
                 root:str,
                 split:str,
                 transform: Optional[Callable]=None,
                 local_radius:float=50)->None:
        super(ArgoverseV1Dataset,self).__init__(root,transform=transform) # 调用要传入root和transform ?why 这两个
        self.root=root
        self._split=split # _只可class内部使用
        self._local_radius=local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names=os.listdir(self.raw_dir) # 返回root/train/data里的csv文件的list
        self._processed_file_names=[os.path.splitext(f)[0]+'.pt' for f in self.raw_file_names] # 保留上面每个文件名，并存成.pt的list
        self._processed_paths=[os.path.join(self.processed_dir,f) for f in self._processed_file_names] # ?why 存这个
        
    # 下面的都是为了区别于torch_geometric.data内的函数的特别写的自定义的，保护文件夹不被外部修改,同时简化get某个self._变量的过程，简单
    @property
    def raw_dir(self)->str:
        return os.path.join(self.root,self._directory,'data')

    @property
    def processed_dir(self)->str:
        return os.path.joni(self.root,self._directory,'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        '''
        要和torch_geometric.data里面自带的processed_file_names区别开来
        raise NotImplementedError
        '''
        return self._processed_file_names

    @property 
    def processed_paths(self) -> List[str]:
        '''
        要和torch_geometric.data里面自带的processed_paths区别开来
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]
        '''
        return self._processed_paths 

    # method1: process并按想要的格式存起来
    def process(self)->None:
        am=ArgoverseMap()
        ''' raw_paths是torch_geometric.data自带的，就是遍历每个csv每辆车！
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]
        '''
        for raw_path in tqdm(self.raw_paths):
            kwargs=process_argoverse(self._split,raw_path,am,self._local_radius) # 
            data=TemporalData(**kwargs) # 以字典形式传入kwargs，?why temporaldata要放进utils.py里面
            torch.save(data,os.path.join(self.processed_dir,str(kwargs['seq_id'])+'.pt'))
    
    # method2:常规的dataset的method——取len
    def len(self)->int:
        return len(self._raw_file_names)
    
    # method3:常规的dataset的method——取元素get
    def get(self,idx)->Data:
        return torch.load(self.processed_paths[idx]) # 用的是自定义的processed_paths，非torch_geometric.data自带的

# 核心！！不应该包进ArgoverseV1Dataset类里！
def process_argoverse(split:str,
                      raw_path:str,
                      am:ArgoverseMap,
                      radius:float)->Dict:
    df=pd.read_csv(raw_path) # Dataframe格式所以df
    
    # step 1: filter out the info of each vehicle (one csv is one vehicle) within the historical/observed (20 frames) period
    # 取出每个车csv所有前20帧的行histrical_df； 所有在前20帧里无重复track_ID的行df 两个不一样 # ?why
    timestamps=list(np.sort(df['TIMESTAMP'].unique())) #list:一辆车被记录的所有帧，df['TIMESTAMP']是Series, Hash table-based unique去重以后unique()返回ndarray
    historical_timestamps=timestamps[:20] # ?why 可改 只取前20帧从0-19
    historical_df=df[df['TIMESTAMP'].isin(historical_timestamps)]   # 返回boolean series，然后选择true的行

    actor_ids=list(historical_df['TRACK_ID'].unique()) # 前20帧的无重复的veh_id
    df=df[df['TRACK_ID'].isin(actor_ids)] # 是series
    num_nodes=len(actor_ids) # actor_ids是车ID

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc # 返回pandas.index类 是class
    av_index=actor_ids.index(av_df[0]['TRACK_ID']) # 取自车av_df出现的第一个track_ID的在actor_id里的index,track_ID是veh_ID
    agent_df=df[df['OBJECT_TYPE']=='AGENT'].iloc
    agent_index=actor_ids.index(agent_df[0]['TRACK_ID'])
    city=df['CITY_NAME'].values # 第一个 因为一张表里都在一个城市

    # step 2: 自己想不到make the coordinates of the scene from global to local (the origin of the coordinates is (X,Y) of AV  19th frame)
    origin=torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float) # 每个csv车里面默认筛选好了只有50行av和agent
    av_heading_vector=origin-torch.tensor([av_df[18]['X'],av_df[18]['Y']],dtype=torch.float)
    theta=torch.atan2(av_heading_vector[1],av_heading_vector[0]) # arctan(y/x)
    rotate_mat=torch.tensor([[torch.cos(theta), -torch.sin(theta)],[torch.sin(theta), torch.cos(theta)]]) # 见word推导
    
    # step 3: initialization——create null tensor to store info:  padding_mask,相对坐标x
    x=torch.zeros(num_nodes,50,2,dtype=torch.float) # 输入x：[num_nodes辆AV自车, 50帧=20+30过去+预测,2被预测的xy坐标]
    edge_index=torch.LongTensor(list(permutations(range(num_nodes),2))).t().contiguous()  # GNN的edge，从N=num_nodes挑出两个表示起点终点 即PN2=N x (N-1)
    padding_mask=torch.ones(num_nodes,50,dtype=torch.bool) # 为了补齐长度有的不够50帧; torch.bool!! 默认是1 true要补值
    bos_mask=torch.zeros(num_nodes,20,dtype=torch.bool) # bos: begin of setences ?why
    rotate_angles=torch.zeros(num_nodes,dtype=torch.float)

    for actor_id,actor_df in df.groupby('TRACK_ID'):
        node_idx=actor_ids.index(actor_id) # return first index of actor_id
        node_steps=[timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']] # 遍历所有vehicle的所有帧的位置index号
        padding_mask[node_idx,node_steps]=False # node_idx辆车以及node_steps帧时间 有信息，不需要padding补值，所以False
        if padding_mask[node_idx,19]: #  make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx,20:]=True  # 历史帧不够长，不预测,全部补值
        
        xy=torch.from_numpy(np.stack([actor_df['X'].values,actor_df['Y'].values],axis=-1))
        x[node_idx,node_steps]=torch.matmul(xy-origin,rotate_mat) # 相对坐标 并旋转！！
        node_historical_steps=list(filter(lambda node_step:node_step<20,node_steps))

        if len(node_historical_steps)>1:  # calculate the heading of the actor (approximately)
            heading_vector=x[node_idx,node_historical_steps[-1]]-x[node_idx,node_historical_steps[-2]] # 当前这辆车这帧的朝向角
            rotate_angles[node_idx]=torch.atan2(heading_vector[1],heading_vector[0]) # 设每辆车的角度
        else:
            padding_mask[node_idx,20:]=True # 就没必要预测了没法测 make no predictions for the actor if the number of valid time steps is less than 2
        
    # step 4: 定义输入x[:,:20] 输出x[:,20:]，bos_mask is True if time step t is valid and time step t-1 is invalid  就是对最开始的那一帧设为true要补值，因为其他都至少长度是2
    bos_mask[:, 0] = ~padding_mask[:, 0] # bos是某辆车的0帧，第一维是veh_ID，第二维是time_step,begin of sentence  如果padding是true那么头一帧都没，肯定bos句子没开头，所以取反 false
    bos_mask[:,1:20]= padding_mask[:, : 19] & ~padding_mask[:, 1: 20] # 妙啊！！！用历史帧t-1 0-18和当前帧t 1-19比， 如果padding_mask是true 则invalid, 若false则valid

    positions=x.clone() # ?why

     # 就是未来的30帧，有当前帧但未来不全或者未来全的，要看前面padding_mask怎么求的 推测出这个条件的
    x[:,20:]=torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1), # 这个维度为什么怪怪的感觉不对啊 ? why
                         torch.zeros(num_nodes,30,2),
                         x[:, 20:] - x[:, 19].unsqueeze(-2)) #　-2因为补上倒数第二维度，单取一个19 和取20：到底的维度不一样，所以要.unsqueeze(-2) 都是相对当前第20帧的坐标 -x[:,19].
    
    # 就是历史的20帧 不含第20个，从第0个-19个
    x[:,1:20]=torch.wher((padding_mask[:,:19]|padding_mask[:,1:20]).unsqueeze(-1),
                         torch.zeros(num_nodes,19,2),
                         x[:,1:20]-x[:,:19])
    x[:,0]=torch.zeros(num_nodes,2)

    # step 5: get lane features at the current time step
    df_19=df[df['TIMESTAMP']== timestamps[19]] 
    node_inds_19= [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']] # 当前帧的所有车
    node_positions_19=torch.from_numpy(np.stack([df_19['X'].values,df_19['Y'].values],axis=-1)).float()# 当前帧的所有车的坐标

    (lane_vectors,is_intersections,turn_directions,traffic_controls,lane_actor_index,lane_actor_vectors)=get_lane_features(am,node_inds_19,node_positions_19,origin,rotate_mat,city,radius)

    y=None if split=='test' else x[:,20:]
    seq_id=os.path.splitext(os.path.basename(raw_path))[0] # 每个csv的编号
    # 返回dict 那么多信息，不只是x,y!!
    return {
        'x': x[:,:20], # [N,20,2] N:num_nodes of GNN= num_vehicles
        'positions': positions, # [N,50,2] :x的克隆 完整的20+30 ?why 不知道什么用
        'edge_index':edge_index, # [2,N x N-1] 从N个节点挑2个组成边！PN2=N x (N-1)；2是起点终点
        'y':y, # [N,30,2]
        
        'num_nodes':num_nodes, #N
        'padding_mask': padding_mask, # [N,50] # 补x成统一长度
        'bos_mask': bos_mask, # [N,20] 判断是否有过去一帧（是补值出来的or本来就有）
        'rotate_angles': rotate_angles, # [N] 是tensor

        'lane_vectors': lane_vectors,  #[L,2]
        'is_intersections': is_intersections, # [L]
        'turn_directions': turn_directions, #[L]
        'traffic_controls':traffic_controls, #[L]
        
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]   # ?why
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]

        'seq_id': int(seq_id),  #?why int 把某辆车的csv文件名的string->int
        'av_index':av_index, # 自车的veh_id
        'agent_index':agent_index, # 感兴趣他车的id ?why只取第一个出现的?
        
        'city':city,
        'origin':origin.unsqueeze(0), # 在dim=0 加一维 维度不清?why 应该是[2] 坐标原点：自车在第20帧就是19的(X,Y)
        'theta':theta, # 自车在第20帧就是19的heading arctan(Δy,Δx) 和上一帧去比！！        
    }

def get_lane_features():
