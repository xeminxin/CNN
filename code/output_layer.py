import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 입력 데이터의 크기
input_size = 10

# 출력 데이터의 크기
output_size = 4

# 출력층 정의
output_layer = nn.Linear(input_size, output_size)

# 가중치 행렬 추출
weights = output_layer.weight.detach().numpy()

# 가중치 행렬 시각화
plt.figure(figsize=(10, 4))
plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel('Input Features')
plt.ylabel('Output Units')
plt.title('Output Units')
plt.title('Output layer Weights')
plt.colorbar()
plt.show()