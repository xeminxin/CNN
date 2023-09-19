import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 입력 데이터의 크기
input_size = 4

# 출력 데이터의 크기
output_size = 2

# 입력 크기 4, 출력 크기 2 (임의 지정)


# Dense Layer 정의
dense_layer = nn.Linear(input_size, output_size)


# 가중치 행렬 추출
weights = dense_layer.weight.detach().numpy()


# 가중치 행렬 시각화
plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel('input features')   # 입력 요소 (4개)
plt.ylabel('Output units')  # 출력 요소 (2개)
plt.title('Dense Layer Weights')
plt.colorbar()
plt.show()