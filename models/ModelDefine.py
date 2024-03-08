import torch.nn as nn
import torchvision
import sys
from models.SwinTransformer import *
from models.build import *
from models.MedViT import *

def NET(config):

    if config.MODEL.TYPE == 'resnet50':
        
        class ResNet50(nn.Module):
            def __init__(self):
                super(ResNet50, self).__init__()
                self.model = build_model(config)
                
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                out = self.model(x)
                out = self.softmax(out)  # クラスタ数の次元を持った確率ベクトル

                return out
            
            def forward_feature(self, x):
                out = self.model(x)

                return out
            
            
            
        net = ResNet50()
        
        return net
    
    if config.MODEL.TYPE == 'swin':
        
        class Swin_Tiny(nn.Module):
            def __init__(self):
                super(Swin_Tiny, self).__init__()
                
                self.model = build_model(config)
                
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                x = self.model(x)
                out = self.softmax(x)
                
                return out
            
            def forward_feature(self, x):
                out = self.model(x)

                return out
            
            
        net = Swin_Tiny()
        
        return net  
    
    if config.MODEL.TYPE == 'medvit':
        
        class Med_small(nn.Module):
            def __init__(self):
                super(Med_small, self).__init__()
                self.model = build_model(config)
                
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                out = self.model(x)
                out = self.softmax(out)  # クラスタ数の次元を持った確率ベクトル

                return out  
            
        net = Med_small()  
        
        return net          
    
    else:
        print('model is not defined')
        sys.exit()