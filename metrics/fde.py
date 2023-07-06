from typing import Any,Callable,Optional
import torch
from torchmetrics import Metric

class FDE(Metric):
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(FDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred:torch.Tensor,
               target:torch.Tensor)->None:
        self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum() # 因为求average，所以先取sum 再/count个数
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
