# '''
# 见argoverse-api/demo usage的argoverse_forecasting_tutorial.ipynb 
# visualizing sequences的viz_sequence源码 抄的这里
# author在issue回复还参考了lstm baseline的utils.py https://github.com/jagjeet-singh/argoverse-forecasting
# '''
# from argoverse.visualization.visualize_sequences import viz_sequence
# import matplotlib as plt
# import numpy as np
# from argoverse.map_representation.map_api import ArgoverseMap
# from collections import OrderedDict
# from typing import Dict,List,Union
# import pandas as pd


# seq_path = f"{root_dir}/2645.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)
# seq_path = f"{root_dir}/3828.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)

# def plot_single_vehicle(
#     avm: ArgoverseMap, 
#     sample_past_trajectory: np.ndarray,  # (20, 2)
#     sample_groundtruth: np.ndarray,  # (20, 2) 
#     sample_forecasted_trajectories: List[np.ndarray],  # List[(30, 2)]
#     sample_city_name: str, 
#     ls: str):

#     sample_groundtruth = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_groundtruth), axis=0)
#     for sample_forecasted_trajectory in sample_forecasted_trajectories:
#         sample_forecasted_trajectory = np.concatenate((np.expand_dims(sample_past_trajectory[-1], axis=0), sample_forecasted_trajectory), axis=0)

#     ## Plot history
#     obs_len = sample_past_trajectory.shape[0]
#     pred_len = sample_groundtruth.shape[0]
#     plt.plot(
#         sample_past_trajectory[:, 0],
#         sample_past_trajectory[:, 1],
#         color="#ECA154",
#         label="Past Trajectory",
#         alpha=1,
#         linewidth=3,
#         zorder=15,
#         ls = ls
#     )

#     ## Plot future
#     plt.plot(
#         sample_groundtruth[:, 0],
#         sample_groundtruth[:, 1],
#         color="#d33e4c",
#         label="Ground Truth",
#         alpha=1,
#         linewidth=3.0,
#         zorder=20,
#         ls = "--"
#     )

#     ## Plot prediction
#     for j in range(len(sample_forecasted_trajectories)):
#         plt.plot(
#             sample_forecasted_trajectories[j][:, 0],
#             sample_forecasted_trajectories[j][:, 1],
#             color="#007672",
#             label="Forecasted Trajectory",
#             alpha=1,
#             linewidth=3.2,
#             zorder=15,
#             ls = "--"
#         )
        
#         # Plot the end marker for forcasted trajectories
#         plt.arrow(
#             sample_forecasted_trajectories[j][-2, 0], 
#             sample_forecasted_trajectories[j][-2, 1],
#             sample_forecasted_trajectories[j][-1, 0] - sample_forecasted_trajectories[j][-2, 0],
#             sample_forecasted_trajectories[j][-1, 1] - sample_forecasted_trajectories[j][-2, 1],
#             color="#007672",
#             label="Forecasted Trajectory",
#             alpha=1,
#             linewidth=3.2,
#             zorder=15,
#             head_width=1.1,
#         )
    
#     ## Plot the end marker for history
#     plt.arrow(
#             sample_past_trajectory[-2, 0], 
#             sample_past_trajectory[-2, 1],
#             sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
#             sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
#             color="#ECA154",
#             label="Past Trajectory",
#             alpha=1,
#             linewidth=3,
#             zorder=25,
#             head_width=1.0,
#         )

#     ## Plot the end marker for future
#     plt.arrow(
#             sample_groundtruth[-2, 0], 
#             sample_groundtruth[-2, 1],
#             sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
#             sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
#             color="#d33e4c",
#             label="Ground Truth",
#             alpha=1,
#             linewidth=3.0,
#             zorder=25,
#             head_width=1.0,
#         )

#     ## Plot history context
#     for j in range(obs_len):
#         lane_ids = avm.get_lane_ids_in_xy_bbox(
#             sample_past_trajectory[j, 0],
#             sample_past_trajectory[j, 1],
#             sample_city_name,
#             query_search_range_manhattan=65,
#         )
#         [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]

#     ## Plot future context
#     for j in range(pred_len):
#         lane_ids = avm.get_lane_ids_in_xy_bbox(
#             sample_groundtruth[j, 0],
#             sample_groundtruth[j, 1],
#             sample_city_name,
#             query_search_range_manhattan=65,
#         )
#         [avm.draw_lane(lane_id, sample_city_name) for lane_id in lane_ids]



# def viz_predictions(
#         input_: np.ndarray,
#         output: np.ndarray,
#         target: np.ndarray,
#         centerlines: np.ndarray,
#         city_names: np.ndarray,
#         idx=None,
#         show: bool = True,
# ) -> None:
#     """Visualize predicted trjectories.

#     Args:
#         input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
#         output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
#         target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
#         centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
#         city_names (numpy array): city names for each trajectory
#         show (bool): if True, show

#     """
#     num_tracks = input_.shape[0]
#     obs_len = input_.shape[1]
#     pred_len = target.shape[1]

#     plt.figure(0, figsize=(8, 7))
#     avm = ArgoverseMap()
#     for i in range(num_tracks):
#         plt.plot(
#             input_[i, :, 0],
#             input_[i, :, 1],
#             color="#ECA154",
#             label="Observed",
#             alpha=1,
#             linewidth=3,
#             zorder=15,
#         )
#         plt.plot(
#             input_[i, -1, 0],
#             input_[i, -1, 1],
#             "o",
#             color="#ECA154",
#             label="Observed",
#             alpha=1,
#             linewidth=3,
#             zorder=15,
#             markersize=9,
#         )
#         plt.plot(
#             target[i, :, 0],
#             target[i, :, 1],
#             color="#d33e4c",
#             label="Target",
#             alpha=1,
#             linewidth=3,
#             zorder=20,
#         )
#         plt.plot(
#             target[i, -1, 0],
#             target[i, -1, 1],
#             "o",
#             color="#d33e4c",
#             label="Target",
#             alpha=1,
#             linewidth=3,
#             zorder=20,
#             markersize=9,
#         )

#         for j in range(len(centerlines[i])):
#             plt.plot(
#                 centerlines[i][j][:, 0],
#                 centerlines[i][j][:, 1],
#                 "--",
#                 color="grey",
#                 alpha=1,
#                 linewidth=1,
#                 zorder=0,
#             )

#         for j in range(len(output[i])):
#             plt.plot(
#                 output[i][j][:, 0],
#                 output[i][j][:, 1],
#                 color="#007672",
#                 label="Predicted",
#                 alpha=1,
#                 linewidth=3,
#                 zorder=15,
#             )
#             plt.plot(
#                 output[i][j][-1, 0],
#                 output[i][j][-1, 1],
#                 "o",
#                 color="#007672",
#                 label="Predicted",
#                 alpha=1,
#                 linewidth=3,
#                 zorder=15,
#                 markersize=9,
#             )
#             for k in range(pred_len):
#                 lane_ids = avm.get_lane_ids_in_xy_bbox(
#                     output[i][j][k, 0],
#                     output[i][j][k, 1],
#                     city_names[i],
#                     query_search_range_manhattan=2.5,
#                 )

#         for j in range(obs_len):
#             lane_ids = avm.get_lane_ids_in_xy_bbox(
#                 input_[i, j, 0],
#                 input_[i, j, 1],
#                 city_names[i],
#                 query_search_range_manhattan=2.5,
#             )
#             [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
#         for j in range(pred_len):
#             lane_ids = avm.get_lane_ids_in_xy_bbox(
#                 target[i, j, 0],
#                 target[i, j, 1],
#                 city_names[i],
#                 query_search_range_manhattan=2.5,
#             )
#             [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

#         plt.axis("equal")
#         plt.xticks([])
#         plt.yticks([])
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = OrderedDict(zip(labels, handles))
#         if show:
#             plt.show()


# def viz_predictions_helper(
#         forecasted_trajectories: Dict[int, List[np.ndarray]],
#         gt_trajectories: Dict[int, np.ndarray],
#         features_df: pd.DataFrame,
#         viz_seq_id: Union[None, List[int]],
# ) -> None:
#     """Visualize predictions.

#     Args:
#         forecasted_trajectories: Trajectories forecasted by the algorithm.
#         gt_trajectories: Ground Truth trajectories.
#         features_df: DataFrame containing the features
#         viz_seq_id: Sequence ids to be visualized

#     """
#     args = parse_arguments()
#     seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
#     for seq_id in seq_ids:
#         gt_trajectory = gt_trajectories[seq_id]
#         curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
#         input_trajectory = (
#             curr_features_df["FEATURES"].values[0]
#             [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
#                 "float"))
#         output_trajectories = forecasted_trajectories[seq_id]
#         candidate_centerlines = curr_features_df[
#             "CANDIDATE_CENTERLINES"].values[0]
#         city_name = curr_features_df["FEATURES"].values[0][
#             0, FEATURE_FORMAT["CITY_NAME"]]

#         gt_trajectory = np.expand_dims(gt_trajectory, 0)
#         input_trajectory = np.expand_dims(input_trajectory, 0)
#         output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
#         candidate_centerlines = np.expand_dims(np.array(candidate_centerlines), 0)
#         city_name = np.array([city_name])
#         viz_predictions(
#             input_trajectory,
#             output_trajectories,
#             gt_trajectory,
#             candidate_centerlines,
#             city_name,
#             show=True,
#         )


'''
python visualize.py --ckpt /mnt/home/husiyuan/code/argo_my/ARGO1/baseline/HiVT/xiaohu_Argo/lightning_logs/version_2/checkpoints/epoch=63-step=102975.ckpt
'''
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
import torch
from pytorch_lightning.callbacks import Callback
from argoDataloader import ArgoverseV1Dataloader
from models.xiaohuNet import HiVT 


class SaveCallback(Callback):
    def __init__(self) -> None:
        self.ckpt_path = "h5/"
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        super().__init__()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.result["y_hat_agent"] = []
        pl_module.result["y_agent"] = []
        pl_module.result["y_hat_agent_ev"] = []
        pl_module.result["y_agent_ev"] = []

        # stupid api here!
        pl_module.save_pred = True
        pl_module.vis = False

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        torch.save(pl_module.result, self.ckpt_path + "result.pt")


if __name__ == "__main__":

    pl.seed_everything(2023)  # fix randomness

    model_name = "HiVT"
    # Model = get_model(model_name)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/home/husiyuan/code/argo_my/ARGO1/data/test_obs/data")
    parser.add_argument("--ckpt", type=str, required=True)
    # set to 1 to test the time for single sample inference
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--persistent_workers", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)  # num of gpu
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--monitor", type=str, default="val_minFDE", choices=["val_minADE", "val_minFDE", "val_minMR"])
    parser.add_argument("--save_top_k", type=int, default=6)
    parser.add_argument("--local_radius", type=int, default=50)
    parser.add_argument("--enable_model_summary", type=lambda x: x.lower() == "true", default=True)
    # parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    data_loader = ArgoverseV1Dataloader.from_argparse_args(args)

    model = HiVT.load_from_checkpoint(args.ckpt)  # strict=False, pred_p=False, embed="LSTM", huber=True, drop_actor=False)

    trainer = pl.Trainer.from_argparse_args(args)  # callbacks=[SaveCallback()])

    trainer.predict(model, data_loader)
