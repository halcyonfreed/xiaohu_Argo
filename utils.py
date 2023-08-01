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
from typing import Any, List,Optional,Tuple # 检查type的error
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
        
class DistanceDropEdge(object):
    '''
    我不理解, 形参是个object, very rarely seen
    因为是要把 一个类实例 变成 可调用的对象  class->object 用object+__call__
    '''
    def __init__(self,max_distance:Optional[float]=None) -> None:
        self.max_distance=max_distance
        
    def __call__(self, 
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m:nn.Module)->None:
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
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)   
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    
    # decoder出现
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

    # xiaohunet出现
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])


class Visualizer():
    def __init__(self) -> None:
    
    def plot_single_vehicle(
        avm: ArgoverseMap, 
        sample_past_trajectory: np.ndarray,  # (20, 2)
        sample_groundtruth: np.ndarray,  # (20, 2) 
        sample_forecasted_trajectories: List[np.ndarray],  # List[(30, 2)]
        sample_city_name: str, 
        ls: str):

        sample_groundtruth = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_groundtruth), axis=0)
        for sample_forecasted_trajectory in sample_forecasted_trajectories:
            sample_forecasted_trajectory = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_forecasted_trajectory), axis=0)

        ## Plot history
        obs_len = sample_past_trajectory.shape[0]
        pred_len = sample_groundtruth.shape[0]
        plt.plot(
            sample_past_trajectory[:, 0],
            sample_past_trajectory[:, 1],
            color="#ECA154",
            label="Past Trajectory",
            alpha=1,
            linewidth=3,
            zorder=15,
            ls = ls
        )

        ## Plot future
        plt.plot(
            sample_groundtruth[:, 0],
            sample_groundtruth[:, 1],
            color="#d33e4c",
            label="Ground Truth",
            alpha=1,
            linewidth=3.0,
            zorder=20,
            ls = "--"
        )

        ## Plot prediction
        for j in range(len(sample_forecasted_trajectories)):
            plt.plot(
                sample_forecasted_trajectories[j][:, 0],
                sample_forecasted_trajectories[j][:, 1],
                color="#007672",
                label="Forecasted Trajectory",
                alpha=1,
                linewidth=3.2,
                zorder=15,
                ls = "--"
            )
            
            # Plot the end marker for forcasted trajectories
            plt.arrow(
                sample_forecasted_trajectories[j][-2, 0], 
                sample_forecasted_trajectories[j][-2, 1],
                sample_forecasted_trajectories[j][-1, 0] - sample_forecasted_trajectories[j][-2, 0],
                sample_forecasted_trajectories[j][-1, 1] - sample_forecasted_trajectories[j][-2, 1],
                color="#007672",
                label="Forecasted Trajectory",
                alpha=1,
                linewidth=3.2,
                zorder=15,
                head_width=1.1,
            )
        
        ## Plot the end marker for history
        plt.arrow(
                sample_past_trajectory[-2, 0], 
                sample_past_trajectory[-2, 1],
                sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
                sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
                color="#ECA154",
                label="Past Trajectory",
                alpha=1,
                linewidth=3,
                zorder=25,
                head_width=1.0,
            )

        ## Plot the end marker for future
        plt.arrow(
                sample_groundtruth[-2, 0], 
                sample_groundtruth[-2, 1],
                sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
                sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
                color="#d33e4c",
                label="Ground Truth",
                alpha=1,
                linewidth=3.0,
                zorder=25,
                head_width=1.0,
            )

        ## Plot history context
        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                sample_past_trajectory[j, 0],
                sample_past_trajectory[j, 1],
                sample_city_name,
                query_search_range_manhattan=65,
            )
            [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]

        ## Plot future context
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                sample_groundtruth[j, 0],
                sample_groundtruth[j, 1],
                sample_city_name,
                query_search_range_manhattan=65,
            )
            [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]


    def viz_predictions(
            input_: np.ndarray,
            output: np.ndarray,
            target: np.ndarray,
            centerlines: np.ndarray,
            city_names: np.ndarray,
            idx=None,
            show: bool = True,
    ) -> None:
        """Visualize predicted trjectories.

        Args:
            input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
            output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
            target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
            centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
            city_names (numpy array): city names for each trajectory
            show (bool): if True, show

        """
        num_tracks = input_.shape[0]
        obs_len = input_.shape[1]
        pred_len = target.shape[1]

        plt.figure(0, figsize=(8, 7))
        avm = ArgoverseMap()
        for i in range(num_tracks):
            plt.plot(
                input_[i, :, 0],
                input_[i, :, 1],
                color="#ECA154",
                label="Observed",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                input_[i, -1, 0],
                input_[i, -1, 1],
                "o",
                color="#ECA154",
                label="Observed",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            plt.plot(
                target[i, :, 0],
                target[i, :, 1],
                color="#d33e4c",
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=20,
            )
            plt.plot(
                target[i, -1, 0],
                target[i, -1, 1],
                "o",
                color="#d33e4c",
                label="Target",
                alpha=1,
                linewidth=3,
                zorder=20,
                markersize=9,
            )

            for j in range(len(centerlines[i])):
                plt.plot(
                    centerlines[i][j][:, 0],
                    centerlines[i][j][:, 1],
                    "--",
                    color="grey",
                    alpha=1,
                    linewidth=1,
                    zorder=0,
                )

            for j in range(len(output[i])):
                plt.plot(
                    output[i][j][:, 0],
                    output[i][j][:, 1],
                    color="#007672",
                    label="Predicted",
                    alpha=1,
                    linewidth=3,
                    zorder=15,
                )
                plt.plot(
                    output[i][j][-1, 0],
                    output[i][j][-1, 1],
                    "o",
                    color="#007672",
                    label="Predicted",
                    alpha=1,
                    linewidth=3,
                    zorder=15,
                    markersize=9,
                )
                for k in range(pred_len):
                    lane_ids = avm.get_lane_ids_in_xy_bbox(
                        output[i][j][k, 0],
                        output[i][j][k, 1],
                        city_names[i],
                        query_search_range_manhattan=2.5,
                    )

            for j in range(obs_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    input_[i, j, 0],
                    input_[i, j, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )
                [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
            for j in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    target[i, j, 0],
                    target[i, j, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )
                [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

            plt.axis("equal")
            plt.xticks([])
            plt.yticks([])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if show:
                plt.show()


    def viz_predictions_helper(
            forecasted_trajectories: Dict[int, List[np.ndarray]],
            gt_trajectories: Dict[int, np.ndarray],
            features_df: pd.DataFrame,
            viz_seq_id: Union[None, List[int]],
    ) -> None:
        """Visualize predictions.

        Args:
            forecasted_trajectories: Trajectories forecasted by the algorithm.
            gt_trajectories: Ground Truth trajectories.
            features_df: DataFrame containing the features
            viz_seq_id: Sequence ids to be visualized

        """
        args = parse_arguments()
        seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
        for seq_id in seq_ids:
            gt_trajectory = gt_trajectories[seq_id]
            curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
            input_trajectory = (
                curr_features_df["FEATURES"].values[0]
                [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
                    "float"))
            output_trajectories = forecasted_trajectories[seq_id]
            candidate_centerlines = curr_features_df[
                "CANDIDATE_CENTERLINES"].values[0]
            city_name = curr_features_df["FEATURES"].values[0][
                0, FEATURE_FORMAT["CITY_NAME"]]

            gt_trajectory = np.expand_dims(gt_trajectory, 0)
            input_trajectory = np.expand_dims(input_trajectory, 0)
            output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
            candidate_centerlines = np.expand_dims(np.array(candidate_centerlines), 0)
            city_name = np.array([city_name])
            viz_predictions(
                input_trajectory,
                output_trajectories,
                gt_trajectory,
                candidate_centerlines,
                city_name,
                show=True,
            )

