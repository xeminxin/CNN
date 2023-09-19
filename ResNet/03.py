import torch
import torch.nn as nn 

# ResNet의 기본 정의

class BasicBlock(nn.Module) : 
    
    expansion = 1 # 확장 비율 변수 => ResNet block 채널수를 확장하는 경우에 필요에 의해서 숫자를 늘려주시면됩니다. (기본 1)
    
    def __init__(self, in_channels, out_channels, stride=1) : 
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        """
        BatchNorm2d() -> 딥러닝에서 모델 학습시 안정적인 학습 하기 위한 기법 -> gradient vanishing 문제를 해결 가능 
        입력의 각 채널에 대해서 평균, 분산 계산하고 정규화된 출력을 생성
        """
        self.relu = nn.ReLU(inplace=True)
        """
        in_channels => 입력 특징 맵의 채널 수 
        out_channels => 출력 특징 맵의 채널 수 
        kernel_size => 컨불루션 커널의 크기 
        stride => 컨불루션 보폭 크기 
        bias => 편향값을 사용할지에 대한 여부 
        
        padding = 1 ===> 입력과 출력의 공간적인 크기를 보존 가능 
        bias=True -> 편향값을 컨불루션 레이어에 값을 추가 -> 모델 좀더 유연하게 학습 시키기 위함 
        """
        
        """
        stride 와 in_channels 1이 아닌경우 와 in_channels 와 self.expansion * out_channels 다른 경우 이 조건이 성립
        in_channels 와 self.expansion * out_channels 사이의 차이를 보상 하기위해서 1x1 컨불루션 진행 
        =====> 입력의 공간적인 차원 조정 !! 
        """
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential() ### 전차연결(residual connection) -> 초기화 
        
        if stride != 1 or in_channels != self.expansion * out_channels : 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x) :
        residual = x # 전차 구성 하기 위한 변수 
        ########################## =====> 기본 블럭 내에서 컨불루션 과 정규화 거치는 일반적인 연산 구간 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)        
        out = self.conv2(out)
        out = self.bn2(out)
        ########################## =====> 기본 블럭 내에서 컨불루션 과 정규화 거치는 일반적인 연산 구간 
        
        out += self.shortcut(residual) #### x + residual -> 전차 수행 -> 입력과 출력 크기 일정하게 만들어주고 -> 정규화 
        out = self.relu(out)  ###### 정규화 -> relu 적용 
        
        return out
    
    
# ResNet 모델 정의
class ResNet(nn.Module) : 
    def __init__(self, block, layers, num_classes=1000) :
        super(ResNet, self).__init__()
        
        self.in_Channels = 64 
        
        """
        conv -> bn -> 활성함수 -> maxpool
        """
        self.conv1 = nn.Conv2d(3, 64,kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    """
    _make_layer : 앞에 '_' 시작하는것은 내부 메서드 나타냅니다. 
    ==> ResNet 내에서 반복적으로 사용되는 레이어 블록 구성 역활을 수행 합니다. 
    """
    def _make_layer(self, block, out_channels, blocks, stride=1) : 
        layers = [] #### ->>>>> 레이어들 담을 리스트 
        layers.append(block(self.in_Channels, out_channels, stride))
        
        self.in_Channels = out_channels * block.expansion
        
        for _ in range(1, blocks) : 
            # in_channels -> 이전 설정한 출력 채널 / out_channels -> out_channels
            layers.append(block(self.in_Channels, out_channels))
        
        print(layers)
        return nn.Sequential(*layers) ### nn.Sequential() -> 모델 구성하기위한 컨테이너 클래스 : 순차적으로 레이어 추가 
    
    def forward(self, x) :
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
        
def resnet34(num_classes = 1000) :
    return ResNet(BasicBlock, [3,4,6,3], num_classes)


model = resnet34(num_classes = 1000)
inputs = torch.randn(1,3,224,224)
output = model(inputs)
print(output.shape)