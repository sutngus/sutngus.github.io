# ArcFace 손실 함수를 구현하는 파이토치 모듈
# ArcFace는 얼굴 인식과 같은 분류 작업에서 신뢰성 있는 임베딩을 학습하기 위해 제안된 Additive Angular Margin Loss 방식을 구현
# 이를 통해 더 높은 성능의 특성 임베딩을 얻어낼 수 있다.

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

Class ArcFace
    1. def __init__(self, in_dim, out_dim, s, m):
        - s and m are parameters derived from "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
        - Matrix W:
            1) The matrix W has dimensions in_dim x out_dim.
            2) W is initialized using Xavier initialization.
            3) in_dim: Dimensionality of the tensor resulting from flattening the forward pass of VGG19.
            4) out_dim: Number of classes. 분류 문제의 카테고리 수
            
    2. def forward(self, x):
        - the forward pass of the ArcFace model.

'''

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        nn.init.kaiming_uniform_(self.W)
        
    def forward(self, x):
        normalized_x = F.normalize(x, p=2, dim=1) # 입력 x를 L2 정규화 -> 각도 기반의 계산을 수행가능
        normalized_W = F.normalize(self.W, p=2, dim=0)
    
        # 입력과 가중치의 코사인 유사도를 계산
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        
        # Using torch.clamp() to ensure cosine values are within a safe range,
        # preventing potential NaN losses.
        
        # 코사인 유사도를 각도로 변환
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 마진 추가 및 스케일링
        probability = self.s * torch.cos(theta+self.m)
        
        return probability
    # 이 확률 벡터는 ArcFace 로스를 사용할 때 모델이 학습할 분류 결과를 나타낸다.