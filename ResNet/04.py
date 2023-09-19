import torch
import torch.nn as nn
import torch.nn.functional as F

# Incetion Model 블록 구조
class IncetionModule(nn.Module):
    
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool):
        super(IncetionModule, self).__init__()
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        # -> 입력 in_channels -> out_1x1 변환
        
        
        # 3x3 convolution branch
        self.conv3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)   # reduce : 줄이다...
        # -> 목적 : 입력 채널 줄이기 -> 입력 in_channels -> reduce_3x3 변환
        self.conv3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        # -> 입력 : reduce_3x3 -> out_3x3 변환 / 출력 크기를 입력과 동일하게 유지한다.
        
        
        # 5x5 convolution branch
        self.conv5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.conv5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
    
        # Max Pooling branch
        