import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 장치 설정(GPU 또는 CPU)
device = torch.device('Cuda' if torch.cuda.is_available() else 'cpu')

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 입력 채널 수 1, 출력 채널 수 16인 합성곱 레이어
        # 입력 이미지의 크기를 유지하고 ReLU 활성화 함수 적용
        self.relu1 = nn.ReLu()
        # ReLU 활성화 함수로 비선형성을 도입
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 2x2 크기의 최대 풀링 레이어. 입력 이미지의 크기를 절반으로 줄임
        
        """
        왜 2x2 크기의 최대 풀링 레이어를 사용하는가
        최대 풀링 레이어는 입력 이미지의 공간적인 크기를 줄이는 역할을 한다. 입력 이미지의 공간적인 해상도를 절반으로 줄여줌.
        이러한 다운샘플링은 모델이 더 넓은 수용 영역을 가지게하고, 불필요한 세부 정보를 제거하여 모델이 더 강건하고 일반화된 특징을 학습할 수 있도록 도와줌
        또, 파라미터 수와 계산 비용이 감소함.
        """
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(7 * 7 * 32, 10)
        # 7x7크기의 32개 채널을 가진 이미지를 받아 10개의 클래스로 분류하는 완전 연결 레이어
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        self.conv2(x)
        self.relu2(x)
        self.pool2(x)
        x = x.view(x.size(0), -1)
        self.fc(x)
        return x
        
        
# MNIST 데이터셋 로드
train_dataset = MNIST(root='.', train = True, transform=ToTensor(), downloade=True)
test_dataset = MNIST(root='.', train = False, transform=ToTensor())

# 데이터 로더 생성
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 모델 인스턴스 생성 및 장치로 이동
model = CNN().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습 설정
num_epochs = 10

# 시각화를 위한 그리드 설정
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.tight_layout(pad=4.0)
axs = axs.flatten()

# 학습 루프
epoch_losses = []  # epoch 손실을 저장할 리스트
for epoch in range(num_epochs):
    model.train()   # 모델을 학습 모드로 설정
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()   # 그래디언트 초기화
        
        outputs = model(images)   # 모델 예측
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()    # 역전파
        optimizer.step()    # 가중치 업데이트
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # 테스트 루프
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)  # 모델 예측
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100.0 * correct / total
    print(f'Test Accuracy : {accuracy:.2f}%')