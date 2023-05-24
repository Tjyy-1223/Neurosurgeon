import torch
import torch.nn as nn
from typing import Union,List,Dict,Any,cast
from collections import abc
import time
from torchvision import models

""" vgg-model's features based on cfg - dict(str,list) """
""" vgg-16 各层的配置 M代表添加MaxPool2d层 数字代表下一层的channel参数"""
cfg = [
    [64,2],'M',[128,2],'M',[256,3],'M',[512,3],'M',[512,3],'M'  # vgg 16
    # [64,2],'M',[128,2],'M',[256,2],'M',[512,2],'M',[512,2],'M'  # vgg 13
    # [64,1],'M',[128,1],'M',[256,2],'M',[512,2],'M',[512,2],'M'    # vgg 11
]


""" VGG model """
class VGG(nn.Module):
    def __init__(self,feature:nn.Module,num_classes:int = 1000,init_weights:bool = True) -> None:
        super(VGG, self).__init__()
        self.features = feature
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.len1 = len(self.features)
        self.len2 = len(self.classifier)
        if init_weights:
            self._initialize_weights()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def __iter__(self):
        return SentenceIterator(self.features,self.classifier)

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, index):
        try:
            if index < self.len1:
                layer = self.features[index]
            else:
                layer = self.classifier[index - self.len1]
        except IndexError:
            raise StopIteration()
        return layer

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


""" for model's iter """
class SentenceIterator(abc.Iterator):
    def __init__(self,features,classifier):
        self.features = features
        self.classifier = classifier
        self.len1 = len(features)
        self._index = 0

    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]
            else:
                layer = self.classifier[self._index - self.len1]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer

""" 
    vgg-model's features based on cfg 
    cfg - 代表各层参数配置
    batch_norm - 代表是否需要BatchNorm层
"""
def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layer = nn.MaxPool2d(kernel_size=2,stride=2)
            layers.append(layer)
        else:
            out_channels = cast(int,v[0])
            range_epoch = cast(int,v[1])

            config_list = []
            for epoch in range(range_epoch):
                # 选定正确的conv2d模型
                if epoch == range_epoch - 1:
                    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
                    in_channels = out_channels
                else:
                    conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1)
                config_list.append(conv2d)
                if batch_norm:
                    config_list.append(nn.BatchNorm2d(conv2d.out_channels))
                config_list.append(nn.ReLU(inplace=True))
            layer = nn.Sequential(*config_list)
            layers.append(layer)

    return nn.Sequential(*layers)



# cfgs: Dict[str, List[Union[str, int]]] = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#

def _vgg(cfg:List[Union[List,int]], batch_norm: bool, pretrained: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    return model

def vgg16(pretrained: bool = False, **kwargs: Any) -> VGG:
    return _vgg(cfg, batch_norm=False, pretrained = pretrained, **kwargs)


def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> VGG:
    return _vgg(cfg, batch_norm=True, pretrained = pretrained, **kwargs)