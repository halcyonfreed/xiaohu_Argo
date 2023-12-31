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

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from argoDataset import ArgoverseV1Dataset
from torch_geometric.data import DataLoader
from typing import Optional,Callable
from pytorch_lightning import LightningDataModule

class ArgoverseV1Dataloader(LightningDataModule):
    '''
    见库里面的定义和官网
    data split之类的 都是自带的不是自己拍脑袋想的
    '''
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(ArgoverseV1Dataloader, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        '''
        Use this to download and prepare data.
        '''
        ArgoverseV1Dataset(root=self.root,split='train',transform=self.train_transform,local_radius=self.local_radius)
        ArgoverseV1Dataset(root=self.root,split='val',transform=self.train_transform,local_radius=self.local_radius)
    
    def setup(self, stage: Optional[str] = None) -> None:
        '''
        类似prepare_data()
        Called at the beginning of fit (train + validate), validate, test, and predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.
        '''
        self.train_dataset = ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)
        self.val_dataset = ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)



class ArgoverseV1DataloaderTest(LightningDataModule):
    def __init__(
        self,
        root: str,
        val_batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        local_radius: float = 50,
    ) -> None:
        super().__init__()
        self.root = root
        self.val_batch_size = val_batch_size
        self.shuffle = False
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        ArgoverseV1Dataset(self.root, "test", self.val_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.test_dataset = ArgoverseV1Dataset(self.root, "test", self.val_transform, self.local_radius)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
