'''
    @comment:  这里可以大作文章，laplace因为这里GNN用到了Laplace算子
'''

import torch
import torch.nn as nn

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1) # 倒数第一维分成两块 [F,N,H,4]->2个[F,N,H,2]
        scale = scale.clone() # 复制无关联不共享内存，但是梯度关联
        
        with torch.no_grad():
            scale.clamp_(min=self.eps) # 下限1e-6 torch.clamp(input, min, max, out=None)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale # log自然e
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
