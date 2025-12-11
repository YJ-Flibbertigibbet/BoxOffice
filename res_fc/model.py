from torchvision import models
import torch.nn as nn


#=====================resnet=======================
class imgResTrain(nn.Module):
    def __init__(self):
        super().__init__()

        #考虑到样本量，这里选resnet18
        self.resnet=models.resnet18(pretrained=True)

        #输入维数
        numFeatures=self.resnet.fc.in_features
        
        #自定义
        self.resnet.fc=nn.Identity()

        #全连接
        self.regressor=nn.Sequential(
            nn.Linear(numFeatures*4,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )


    def forward(self,x):
        batchSize,num_imgs,c,h,w=x.size()
        x=x.view(batchSize*num_imgs,c,h,w)
        features=self.resnet(x)
        features=features.view(batchSize,num_imgs,-1)
        features=features.view(batchSize,-1)

        #全连接
        output=self.regressor(features)
        return output.squeeze()
    
    
