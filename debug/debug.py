''''
debug python in vscode
https://zhuanlan.zhihu.com/p/560405414
'''
import torch
import debugpy; debugpy.connect(('0.0.0.0',5678))


# debugpy.listen(address=("0.0.0.0", 5678))
# debugpy.wait_for_client()
# breakpoint()
a=torch.rand(10)
b=a+1
print(b)