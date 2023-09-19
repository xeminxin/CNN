import torch
import torch.nn as nn

# CNN 아키텍처 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # nn.Conv2d를 이용하여 합성곱층 정의
        # 첫 번째 합성곱층은 입력 채널 수가 1, 출력 채널 수가 16인 커널 사용, 커널 크기는 3x3, 스트라이드와 패딩은 1
        self.relu = nn.ReLU()
        # nn.ReLU를 사용해 활성화 함수로 ReLU 정의
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        # nn.MaxPool2d를 사용해 폴링층 정의
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 두 번째 합성곱층 정의
        self.fc = nn.Linear(32 * 7 * 7, 10)
        # 완전 연결층 정의
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    # 입력 데이터를 합성곱층과 폴링층을 통과시킨 후, 펼쳐진 후처리된 데이터를 완전 연결층에 입력으로 넣어 최종 출력 계산
    