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

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from argoDataset import ArgoverseV1Dataset
from torch_geometric.data import DataLoader
from typing import Optional,Callable
from pytorch_lightning import LightningDataModule

class ArgovereseV1DataModule(LightningDataModule):
    '''
    见库里面的定义和官网
    '''
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims):
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    def optimizer