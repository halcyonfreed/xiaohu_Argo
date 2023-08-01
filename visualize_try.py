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
    


