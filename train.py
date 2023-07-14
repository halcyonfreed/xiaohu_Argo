# import faulthandler
# faulthandler.enable()
'''
 python train.py --root /mnt/home/husiyuan/code/argo_my/ARGO1/data/ --embed_dim 128
'''

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argoDataloader import ArgoverseV1Dataloader
from models.xiaohuNet import HiVT

if __name__ == '__main__':
    pl.seed_everything(5201314)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=[0,1,2,3])
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    model = HiVT(**vars(args))
    datamodule = ArgoverseV1Dataloader.from_argparse_args(args)
    trainer.fit(model, datamodule)
