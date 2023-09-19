# LeNet-5 실습
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 데이터 다운로드 (CHFAR-10)
### transform
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=False)



# LeNet-5 모델
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)   # 입력 채널, 출력 채널 수, 커널 사이즈
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)   # 입력 채널, 출력 채널 수, 커널 사이즈
        self.fc1 = nn.Linear(64 * 6 * 6, 64)   # 크기 조정
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x) :
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)    # 1차원으로 펼치기
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    
# 모델 학습 및 평가 함수 구현
def train_and_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 학습 loop
    for epoch in range(5) :
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0) :
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 매 200번째 미니 배치마다 조건이 참이 되어 손실이 출력되고 running_loss가 재설정됨
            # 이는 미니배치가 190번째일때마다 출력 및 초기화가 수정됨
            
            if i % 200 == 190 :
                print('[%d, %5d] loss : %.3f' % (epoch + 1, i+1, running_loss / 200))
                running_loss = 0.0
                
    print('Finished Training ... ')
    
    # 모델 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print('Acc >> %.2f %%' % acc)
    
    
# 실행
print('LeNet-5')
model = LeNet()
train_and_eval(model)

# 파라미터수
print('LeNEt', sum(p.numel() for p in model.parameters()))