import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 입력 데이터 생성
input_data = torch.randn(1, 1, 16, 16)
# (배치크기, 채널 수, 높이, 너비)

# 피치워크 컨볼루션 적용
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=2)
# nn.Conv2d를 사용해 피치워크 컨볼루션 적용
# dilation 인수를 사용해 커널 간격 조절할 수 있음
# dilation=2 -> 커널 간격 2
output = conv(input_data)

# 결과 출력
print(output.size())   # 출력 크기 확인

# 입력 데이터 시각화
plt.subplot(1, 2, 1)
plt.imshow(input_data.squeeze(), cmap='gray')
plt.title('Input')

plt.subplot(1, 2, 2)
plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
plt.title('Output')

plt.tight_layout()
plt.show()