'''
这个都是官方的评价标准metrics
losses是train训练时用来评价model的, 不是一回事
loss可以=metrics最好不要
'''
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric

class ADE(Metric):
    '''
    见Metric的源码
    '''
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(ADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        self.sum += torch.norm(pred - target, p=2, dim=-1).mean(dim=-1).sum()  # 差的最后一维的二范数， 最后一维均值，再取sum
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
