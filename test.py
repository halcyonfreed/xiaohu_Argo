'''
长的类似eval.py,几乎一模一样就是数据集改成test
python test.py --root /mnt/home/husiyuan/code/argo_my/ARGO1/data/ --batch_size 32 --ckpt_path /mnt/home/husiyuan/code/argo_my/ARGO1/baseline/HiVT/xiaohu_Argo/lightning_logs/version_2/checkpoints/epoch=63-step=102975.ckpt
python test.py --root /mnt/home/husiyuan/code/argo_my/ARGO1/data/ --ckpt /mnt/home/husiyuan/code/argo_my/ARGO1/baseline/HiVT/xiaohu_Argo/lightning_logs/version_2/checkpoints/epoch=63-step=102975.ckpt --embed_dim 128
'''

from pytorch_lightning.utilities.model_summary import ModelSummary
from argparse import ArgumentParser
from pytorch_lightning.callbacks import Callback
import os
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from argoverse.evaluation.competition_util import generate_forecasting_h5

from argoDataloader import ArgoverseV1Dataloader,ArgoverseV1DataloaderTest
from argoDataset import ArgoverseV1Dataset
from models.xiaohuNet import HiVT

class SubmissionCallback(Callback):
    def __init__(self, filename) -> None:
        self.ckpt_path = "h5/"
        self.filename = filename
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        super().__init__()

    def on_test_epoch_end(self, trainer, pl_module):
        print("Generating h5 file!")
        generate_forecasting_h5(pl_module.result, output_path=self.ckpt_path, filename=self.filename)



if __name__ == '__main__':
    pl.seed_everything(5201314)

    parser = ArgumentParser()
    # parser.add_argument('--root', type=str, required=True)
    # parser.add_argument('--ckpt', type=str, required=True)
    # parser.add_argument('--train_batch_size', type=int, default=32)
    # parser.add_argument('--val_batch_size', type=int, default=32)
    # parser.add_argument('--test_batch_size', type=int, default=32)
    # parser.add_argument('--shuffle', type=bool, default=False)
    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--pin_memory', type=bool, default=True)
    # parser.add_argument('--persistent_workers', type=bool, default=True)
    # parser.add_argument('--gpus', type=int, default=0)
    # parser.add_argument('--max_epochs', type=int, default=1)
    # parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    # parser.add_argument('--save_top_k', type=int, default=5)
    # parser = HiVT.add_model_specific_args(parser)
    # args = parser.parse_args()

    # model = HiVT.load_from_checkpoint(args.ckpt, drop_actor=False)
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[SubmissionCallback(filename="HiVT")])
    # print(ModelSummary(model))
    # print("\nValidate to ensure no bug in model and ckpt \n")
    # # val_dataset = ArgoverseV1Dataloader.from_argparse_args(args)
    # # trainer.validate(model, val_dataset)
    
    # test_dataset = ArgoverseV1DataloaderTest.from_argparse_args(args)
    # trainer.test(model=model, test_dataloaders=test_dataset)


    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1) #[0,1,2,3])
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model_name="HiVT"
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SubmissionCallback(filename=model_name)])
    
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    test_dataset = ArgoverseV1Dataset(root=args.root, split='test', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.test(model, dataloader)
