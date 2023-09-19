import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LR = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10   # MNIST 데이터 셋은 10개의 클래스를 가지고 있습니다.

# transforms 정의
transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                transforms.ToTensor()])

# data set 다운받고 생성하기
train_dataset = datasets.MNIST(root='mnist_data', train==True, transform=transform, download=True)
valid_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform)

# dataloader 정의
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


class LeNet5(nn.Module):
    
    def __init__(self, n_classes) :
        super(LeNet5, self).__init__()
    
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits.probs
    
    def get_accuracy(model, data_loader, device):
        # 전체 data_loader에 대한 예측의 정확도를 계산하는 함수
        
        correct_pred = 0
        n = 0
        
        with torch.no_grad():
            model.eval()
            for X, y_true in data_loader:
                X = X.to(device)
                y_true = y_true.to(device)
                
                _, y_prob = model(X)
                _, predicted_labels = torch.max(y_prob, 1)
                
                n += y_true.size(0)
                correct_pred += (predicted_labels == y_true).sum()
                
        return correct_pred.float() / n
    
    def plot_losses(train_losses, valid_losses):
        # training과 validation loss를 시각화 하는 함수
        
        # plot style을 seaborn으로 설정
        plt.style.use('seaborn')
        
        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        ax.plot(train_losses, color='blue', label='Training loss')
        ax.plot(valid_losses, color='red', label='Validation loss')
        ax.set(title='Loss over epochs',
               xlabel = 'Epoch',
               ylabel = 'Loss')
        ax.legend()
        fig.show()
        
        # plt style을 기본값으로 설정
        plt.style.use('default')
        
    def train(train_loader, model, criterion, optimizer, device):
        # training loop의 training 단계에 대한 함수
        
        model.train()
        running_loss = 0
            
        for X, y_true in train_loader:
            optimizer.zero_grad()
            
            X = X.to(device)
            y_true = y_true.to(device)
            
            # 순전파
            y_hat, _ = model(X)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        return model, optimizer, epoch_loss
    
    def validate(valid_loader, model, criterion, device):
        # training loop의 validation 단계에 대한 함수
        
        model.eval()
        running_loss = 0
        
        for X, y_true in valid_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            
            # 순전파와 손실 기록하기
            y_hat, _ = model(X)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)
            
        epoch_loss = running_loss / len(valid_loader.dataset)
        
        return model, epoch_loss
    
    def trainig_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
        # 전체 training loop을 정의하는 함수
        
        # metrics를 저장하기 위한 객체 설정
        best_loss = 1e10
        train_losses = []
        valid_losses = []
        
        # model 학습하기
        for epoch in range(0, epochs):
            
            # training
            model, optimizer, train_loss = train(train_loader, model, criterion, optimzer, device)
            train_losses.append(train_loss)
            
            # validation
            with torch.no_grad():
                model, valid_loss = validate(valid_loader, model, criterion, device)
                valid_losses.append(valid_loss)
                
            if epoch % print_every == (print_every - 1):
                train_acc = get_accuracy(model, train_loader, device=device)
                valid_acc = get_accuracy(model, valid_loader, device=device)
                
                print(f'{datetime.now().time().replace(microsecond=0)} ---'
                      f'Epoch : {epoch + 1}\t'
                      f'Train loss : {train_loss:.4f}\t'
                      f'Valid loss : {valid_loss:.4f}\t'
                      f'Train accuracy: {100 * train_acc : .2f}\t'
                      f'Valid accuracy : {100 * valid_acc:.2f}')
                
        plot_losses(train_losses, valid_losses)
        
        return model, optimizer, (train_losses, valid_losses)

if __name__ == "__main__" :
    torch.manual_seed(RANDOM_SEED)
    
    model = LeNet5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)
    
        