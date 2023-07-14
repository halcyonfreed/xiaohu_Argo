'''
https://github.com/ZikangZhou/HiVT/issues/31的YueYao-bot的所有回答要看！！
argoverse-api/demo_usage里的forecasting_tutorial和competition_forecasting.ipynb都看看！
对着utils.py的temporal_data和argoversev1dataset.py
ipynb调试
TemporalData(
    agent_index=12, 
    av_index=0, 
    bos_mask=[35, 20], 
    city="PIT", 
    edge_index=[2, 1190], 
    is_intersections=[1215], 
    lane_actor_index=[2, 8408], 
    lane_actor_vectors=[8408, 2], 
    lane_vectors=[1215, 2], 
    origin=[1, 2], 
    padding_mask=[35, 50],
    positions=[35, 50, 2], 
    rotate_angles=[35], 
    seq_id=21842, 
    theta=-2.8257100582122803, 
    traffic_controls=[1215], 
    turn_directions=[1215], 
    x=[35, 20, 2])
'''


import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import permutations, product
from typing import Tuple, List, Dict
import json
from tqdm.auto import tqdm
import torch
from argoverse.map_representation.map_api import ArgoverseMap
# import sys
# module不在同一个目录的, 开头加一个sys.path
# sys.path.append('HiVT')
# sys.path.append('HiVT/datasets')


from models.xiaohuNet import HiVT

from argoDataset import process_argoverse, get_lane_features, ArgoverseV1Dataset
from utils import TemporalData

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.evaluation.competition_util import generate_forecasting_h5


checkpoint_path = "/mnt/home/husiyuan/code/argo_my/ARGO1/baseline/HiVT/xiaohu_Argo/lightning_logs/version_2/checkpoints/epoch=63-step=102975.ckpt"
model = HiVT.load_from_checkpoint(checkpoint_path=checkpoint_path, parallel=False)

split = 'test'
dataset = ArgoverseV1Dataset(root="/mnt/home/husiyuan/code/argo_my/ARGO1/data/", split=split, local_radius=model.hparams.local_radius)

output_all_k6 = {}
probs_all = {}
for i, inp in enumerate(tqdm(dataset)):
    x = inp.x.numpy() #
    # y = inp.y.numpy() # testset没有y
    seq_id = inp.seq_id
    positions = inp.positions.numpy() # positions是补值前的x的clone
    # the location of the ego vehicle at TIMESTAMP=19 自车是原点
    origin = inp.origin.numpy().squeeze() # [1,2]->[2] tensor->numpy
    
    # the index of the focal agent 当前处理的agent车
    agent_index = inp.agent_index
    
    # ego_heading at TIMESTAMP=19  自车的朝向角
    ego_heading = inp.theta.numpy()
    # 当前处理的车（自车or他车）和自车的角度差 弧度制
    ro_angle = inp.rotate_angles[agent_index].numpy()
    
    # Global rotation to align with ego vehicle
    rotate_mat = np.array([
        [np.cos(ego_heading), -np.sin(ego_heading)],
        [np.sin(ego_heading), np.cos(ego_heading)]
    ])
    
    R =  np.array([
                    [np.cos(ro_angle), -np.sin(ro_angle)],
                    [np.sin(ro_angle), np.cos(ro_angle)]
                ])
    

    # we recover the agent trajectory from the inputs, just as a sanity check
    offset = positions[agent_index, 19, :]
    hist = (np.cumsum(-x[agent_index, 20::-1, :], axis=0)[::-1, :] + offset) @ rotate_mat.T + origin
    # fut =  (y[agent_index, :, :] + offset) @ rotate_mat.T + origin
    

    res, res_pi = model(inp)
    agt_res = res[:, agent_index, :, :].detach().cpu().numpy() # [6, num_agents, 30, 2]

    probs = torch.softmax(res_pi[agent_index], dim=0)
    
    agt_res_origin = (agt_res[:, :, :2] @ R.T + offset) @ rotate_mat.T + origin
    
    probs_all[seq_id] = probs.detach().cpu().numpy()
    output_all_k6[seq_id] = agt_res_origin

output_path = 'competition_files/'
generate_forecasting_h5(output_all_k6, output_path, probabilities= probs_all, filename = 'HiVT128') #this might take a while