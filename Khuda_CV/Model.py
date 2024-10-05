from Backbone import VGG19
from ArcFace import ArcFace
import torch.nn as nn

# 얼굴 인식 모델의 Recognizer 클래스를 정의
# VGG19를 백본으로 사용하고 ArcFace를 최종 분류자(손실 함수로 사용되는 클래스)로 적용
class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.ArcFace = ArcFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        # x = x.view(x.size(0), -1) 
        # 3D 텐서 형태(batch_size x 512 x 7 x 7)인 VGG19의 출력을 ArcFace를 적용하기 위해 1D 벡터로 펼쳐주는 것을 명시적으로 나타냄
        x = self.ArcFace(x)
        
        return x