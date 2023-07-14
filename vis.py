import pytorch_lightning as pl


from argparse import ArgumentParser
import os
import torch
from pytorch_lightning.callbacks import Callback
from argoDataloader import ArgoverseV1Dataloader
from model import get_model

# import debugpy

# debugpy.listen(address=("0.0.0.0", 5678))
# debugpy.wait_for_client()
# breakpoint()


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

    model_name = "FGNet"
    Model = get_model(model_name)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/petrelfs/tangxiaqiang/dataset")
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
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--local_radius", type=int, default=50)
    parser.add_argument("--enable_model_summary", type=lambda x: x.lower() == "true", default=True)
    # parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    data_loader = ArgoDataloader.from_argparse_args(args)

    model = Model.load_from_checkpoint(args.ckpt)  # strict=False, pred_p=False, embed="LSTM", huber=True, drop_actor=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
    )  # callbacks=[SaveCallback()])

    trainer.predict(model, data_loader)
