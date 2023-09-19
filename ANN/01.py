import torch
import torch.nn as nn

## Python과 PyTorch를 사용하여 간단한 ANN 구조 구현 

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        input_size : 입력층의 크기
        hidden_size : 은닉층의 킈
        output_size : 출력층의 크기
        '''
        
        super(ANN, self).__init__()
        # super(ANN, self) -> ANN의 상위 클래스를 호출한다고 명시함.
                                # -> 상속 단계가 2단계 이상일 때 의미가 있다.
        # super() -> 부모 클래스가 한 단계밖에 없는 경우 위와 완전히 동일한 역할이 된다.
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.linear(input_size, output_size) -> input_size : 이 층이 입력 받을 크기, output_size : 이 층이 출력할 크기 = (입력층 - 은닉층)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x : 입력값
        out_by_input_hidden = self.fc1(x)
        # nn.Linear(input_size, hidden_size)로 정의된 계층에 x를 통과시킴
        # -> 입력층 - 은닉층 연산 진행
        out_by_relu = self.relu(out_by_input_hidden)
        # 은닉층의 활성화 함수 적용
        out_by_hidden_input = self.fc2(out_by_relu)
        # 은닉층 - 출력층 연산 진행
        return out_by_hidden_input
    
    
if __name__ == "__main__":
    input_size = 784
    hidden_size = 256
    output_size = 10
    # 임의로 지정된 수치
    
    model = ANN(input_size, hidden_size, output_size)
    # 정의한 인공신경망 모델의 생성
    
    criterion = nn.CrossEntropyLoss()
    # 손실 함수
    
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 최적화 알고리즘 정의
    
    inputs = torch.randn(100, input_size) # 임의로 생성된 100 x input_size 입력 데이터
    labels = torch.randint(0, output_size, (100, ))  # 100 크기의 라벨
    
    num_epochs = 10
    # 학습을 진행할 Epoch 횟수
    
    for epoch in range(num_epochs):
        outputs = model(inputs)
        # ANN 클래스 객체인 model을 직접 호출하면 forward 함수를 내부적으로 호출하게 됨
        # 순전파 진행
        
        loss = criterion(outputs, labels)
        # 실제 정답인 labels와 위에서 얻은 outputs 사이의 오차
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 역전파 과정
        # 최적화 알고리즘에 오차를 적용하여 해당 값을 기준으로 내부 가중치를 업데이트 하도록 함
        
        # 로그 출력
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch + 1} / {num_epochs} : Loss {loss.item():.4f}")
            # if 문은 Epoch n회당 1번만 로그를 출력하도록 지정하기 위함
            
            
###### ANN 특징

## 장점
# 복잡한 패턴 학습
# 대용량 데이터 처리
# 병렬 정리
# 유연성
# 전이 학습

## 문제점
# 깊은 신경망에서 발생하는 그래디언트 소실 : 가중치가 업데이트 되는 과정에서 어떤 신호는 약한 신호로 전파가 되면서 어느 순간 은닉층 중에 특정 노드가 0이 될 수도 있음. 은닉층은 한 번 노드가 0이 되면 더 이상 업데이트가 되지 않는다.
# 하이퍼파라미터 조정 : 사용자가 직접 조정하는 수치들을 하이퍼파라미터라고 하는데, (신경망 층 수, batch size, lr, 뉴런 수(가지 수) 등등) 이걸 다 신경 써야 하기 때문에 사용하기 번거롭다. 잘못 조정하면 모델 성능에 부정적인 영향을 미칠 수 있다.
# 설명 가능성 부족
