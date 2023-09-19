import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 각 에포크에서 합성곱 레이어의 가중치, 완전 연결 레이어의 가중치, 손실 그래프 시각화
# 그리드가 추가되었으며, 첫 번째 합성곱 레이어와 두 번째 합성곱 레이어의 가중치, 그리고 완전 연결 레이어의 가중치를 표시 시각화 실습


# 장치 설정(GPU 또는 CPU)
device = torch.device('Cuda' if torch.cuda.is_available() else 'cpu')

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLu()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(7 * 7 * 32, 10)
    
    def forward(self, x):
        x = self.conv1(x)    
        x = self.relu1(x)    
        x = self.pool1(x)    
        x = self.conv2(x)    
        x = self.relu2(x)    
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    # 모델의 순전파 연산 정의. 입력 데이터를 합성곱, 활성화 함수, 풀링을 거쳐 완전 연결 레이어에 전달
    # 최종적으로 10개의 클래스에 대한 예측값 출력
     
        
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
fig, axs = plt.subplots(2, 2, figsize = (10, 8))
fig.tight_layout(pad=4.0)
axs = axs.flatten()

# 학습 루프
epoch_losses = [] # epoch 손실을 저장할 리스트

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

# 첫 번째 합성곱 레이어의 가중치 시각화
if epoch == 0:
    weights = model.conv1.weight.detach().cpu().numpy()
    axs[0].imshow(weights[0, 0], cmap='coolwarm')
    axs[0].set_title('Conv1 Wegihts')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(axs[0].imshow(weights[0, 0], cmap='coolwarm'), cax=cax)

# 두 번째 합성곱 레이어의 가중치 시각화
if epoch == 0:
    weights = model.conv2.weight.detach().cpu().numpy()
    axs[1].imshow(weights[0, 0], cmap='coolwarm')
    axs[1].set_title('Conv2 Wegihts')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(axs[1].imshow(weights[0, 0], cmap='coolwarm'), cax=cax)

# 완전 연결 레이어의 가중치 시각화
if epoch == 0:
    weights = model.fc.weight.detach().cpu().numpy()
    axs[2].imshow(weights[0, 0], cmap='coolwarm')
    axs[2].set_title('FC Wegihts')
    axs[2].set_xlabel('Input Features')
    axs[2].set_ylabel('Output Units')

    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(axs[2].imshow(weights.T, cmap='coolwarm', aspect='auto'), cax=cax)

    # 학습 과정에서의 손실 시각화
    axs[3].plot(range(epoch+1), epoch_losses)
    axs[3].set_title('Training Loss')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Loss')
    
plt.show()
        

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