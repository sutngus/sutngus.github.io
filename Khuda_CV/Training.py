'''

Code written by Hayoung Lee.
Contact email: lhayoung9@khu.ac.kr (Note: I may not check my email regularly due to my mistake.)

Training conducted using TPU v2 on Google Colaboratory.

'''
# Recognizer 모델을 사용해 이미지 분류 작업을 수행하는 전체적인 워크플로우  

import os
from google.colab import drive

# 1. 데이터 로드 및 준비
# Mount Google Drive

drive.mount('/content/drive')

# Set project folder path

project_folder = '/content/drive/MyDrive/Project3'

# Initialize lists to store image paths and labels

image = []
label = []

# Traverse through the project folder to collect image paths and corresponding labels

for subdir, _, files in os.walk(project_folder):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(subdir, file)
            image.append(image_path)
            
            label_name = os.path.basename(subdir)
            label.append(label_name)
            

from torch.utils.data import DataLoader
from Preprocessing import CustomDataset
from sklearn.model_selection import train_test_split 

BATCH_SIZE = 128

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size = 0.33, random_state = 425)

# Create custom datasets and dataloaders

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


'''

Declaration of Model, Optimizer, etc.

1) Epoch: 100
2) Batch size: 128
    - Due to the small size of the dataset, batch size was increased based on professor's advice.
3) Loss Function: CrossEntropy
4) Optimizer: Adam with Learning rate 0.01

'''

# 2. 모델 및 학습 준비

import time
import torch
import torch.nn as nn

# Model.py에서 정의된 Recognizer 클래스를 import하여 가져옴
from Model import Recognizer

EPOCH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer

MODEL = Recognizer().to(DEVICE)
LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr = 0.001)


'''

Function Definitions

1) 주어진 데이터 로더의 전체 손실과 정확도를 계산
   def compute_accuracy_and_loss(device, model, data_loader):
    - Input: device, model, data_loader
    - Output: loss / example_num, correct_num / example_num * 100
        - 'loss / example_num' represents the average loss.
        - 'correct_num / example_num * 100' represents the accuracy percentage.

2) 모델의 가중치를 특정 경로에 저장
   def save_weight(model, path):
    - This function saves the model's weights at the specified path.
    
'''


def compute_accuracy_and_loss(device, model, data_loader):
    loss, example_num, correct_num = 0, 0, 0
    
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.to(device)
        probability = model(image)
        
        #Calculate loss using CrossEntropy
    
        loss += LOSS(probability, label)
        
        #Calculate accuracy
        
        # _, true_index = torch.max(label, 1)
        # CrossEntropyLoss의 경우, label은 클래스 인덱스 값([0, 1, 2, ...]) 형태, 원-핫 인코딩 형태가 아님.
        _, predict_index = torch.max(probability, 1)
        true_index = label  # `label`은 클래스 인덱스이므로 그대로 사용

        example_num += true_index.size(0)
        # correct_num += (true_index == predict_index).sum
        # .sum()은 tensor를 반환함. 정확도를 계산하려면 정수값이 필요하므로 .item()을 통해 정수로 변환
        # 이러한 변경을 통해 compute_accuracy_and_loss 함수는 데이터셋의 정확도와 손실을 올바르게 계산 가능
        correct_num += (true_index == predict_index).sum.item()

        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(data_loader):03d} |'
               f'Loss: {loss:03f}')
        
    return loss/example_num, correct_num/example_num*100


def save_weight(model, path):
    torch.save(model.state_dict(), path)


'''

Visualizing model architecture by using tensorboard Library

'''
# 3.텐서보드 설정 및 모델의 구조 시각화
# 데이터셋에서 샘플 이미지 하나를 추출하여 모델에 전달하고, 이를 텐서보드에 추가함

from torch.utils.tensorboard import SummaryWriter

image_for_visualization, label_for_visualization = train_dataset[0]

writer = SummaryWriter()
writer.add_graph(MODEL, image_for_visualization.unsqueeze(0))


''' 

Training

'''
# 4. 모델 학습 및 평가

start_time = time.time()

for epoch in range(EPOCH):
    MODEL.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        probability = MODEL(image)
        
        loss = LOSS(probability, label)
        
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch: {batch_idx:03d}/{len(train_loader):03d} |'
               f'Loss: {loss:03f}')
        
    MODEL.eval()
    with torch.no_grad():
        train_loss, train_acc = compute_accuracy_and_loss(DEVICE, MODEL, train_loader)
        test_loss, test_acc = compute_accuracy_and_loss(DEVICE, MODEL, test_loader)
        
        # Add scalars to tensorboard for visualization
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        writer.flush()
    
    # Save model weights every 10 epochs
    
    if epoch%10 == 0:
        save_weight(MODEL.VGG19, f"/content/drive/MyDrive/VGG19_{epoch}.pth")
        save_weight(MODEL.ArcFace, f"/content/drive/MyDrive/ArcFace_{epoch}.pth")
        # 모델을 저장할 때마다 드라이브에 파일이 쌓이는 것을 방지하기 위해 기존 모델의 가중치를 덮어쓰거나, 최상의 모델만 저장하는 로직을 추가하기
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs