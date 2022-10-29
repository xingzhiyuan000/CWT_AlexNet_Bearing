import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络
import torch.nn as nn
import torch


class CWT_AlexNet(nn.Module):
    def __init__(self, num_classes=13, init_weights=False):
        super(CWT_AlexNet, self).__init__()
        self.features = nn.Sequential(  #打包
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),  # input[3, 224, 224]  output[96, 54, 54] 自动舍去小数点后
            nn.ReLU(inplace=True), #inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=2, stride=2),                  # output[96, 27, 27]
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),           # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 13, 13]
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # output[256, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=CWT_AlexNet() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,3,224,224)) #生成一个batchsize为64的，1个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(3,224,224)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
