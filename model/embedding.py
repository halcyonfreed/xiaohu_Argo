import torch
import torch.nn as nn
from utils import init_weights
from typing import List,Optional

'''
所谓embedding 就是堆了很多层的linear norm relu 
that's it
'''

class SingleInputEmbedding(nn.Module):
    '''
    singleInput 对自车central vehicle
    '''
    def __init__(self,
                 in_channel:int,
                 out_channel:int) -> None:
        super(SingleInputEmbedding,self).__init__()
        self.embed=nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),

            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),

            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        # .apply(): Applies fn(init weights ) recursively to every submodule (as returned by .children()) as well as self.
        self.apply(init_weights) # ?why 为什么搞个init weights 对不同layer参数里的权重w设置不同的初值
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.embed(x)
    
class MultipleInputEmbedding(nn.Module):
    '''
    multipleInput: nbr是自车的半径内 感兴趣范围内的neighbour vehicles
    '''
    def __init__(self,
                 in_channels:List[int],
                 out_channel:int)->None:
        super(MultipleInputEmbedding,self).__init__()
        
        # 不分顺序的list，成员是nn.sequential的模块
        self.module_list=nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel)) 
                           for in_channel in in_channels])
        
        # 然后再 分顺序
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i]=self.module_list[i](continuous_inputs[i]) # ?why没懂
        output=torch.stack(continuous_inputs).sum(dim=0) #?why 在dim=0上连接并在dim0求和

        if categorical_inputs is not None:
            output+=torch.stack(categorical_inputs).sum(dim=0)
        
        return self.aggr_embed(output)