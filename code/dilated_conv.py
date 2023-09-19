import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms

# 이미지 로드
image_path = "./surprised_cat.jpg"
image = Image.open(image_path).convert('L')  # 그레이스케일로 변환
input_data = transforms.ToTensor()(image).unsqueeze(0)  # (배치 크기, 채널 수, 높이, 너비)

# 피치워크 컨볼루션 적용
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2)
output = conv(input_data)


# 입력 이미지 시각화
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('input image')

# 피치워크 컨볼루션 결과 시각화
plt.subplot(1, 2, 2)
plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
plt.title('output image')

plt.title_layout()
plt.show()