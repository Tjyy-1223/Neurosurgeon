import torch
import torch.nn as nn
from collections import abc


class LeNet(nn.Module):
    def __init__(self,input_channels=3,num_classes: int = 1000) -> None:
        """
            input_channels: 输入图像的通道数，默认通道数为3
            num_classes: LeNet的输出维度，默认为1000
        """
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=(5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 54 * 54, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        self.len1 = len(self.features)
        self.len2 = len(self.classifier)
        self.len = self.len1 + self.len2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def __iter__(self,):
        """ 用于遍历LeNet模型的每一层 """
        return SentenceIterator(self.features,self.classifier)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        try:
            if item < self.len1:
                layer = self.features[item]
            else:
                layer = self.classifier[item - self.len1]
        except IndexError:
            raise StopIteration()
        return layer



class SentenceIterator(abc.Iterator):
    """
    LeNet迭代器
    下面是 LeNet 网络的迭代参数调整
    将下面的设置传入到 LeNet 的 __iter__ 中可以完成对于 LeNet 网络的层级遍历
    """
    def __init__(self,features,classifier):
        self.features = features
        self.classifier = classifier
        self._index = 0
        self.len1 = len(features)
        # self.len2 = len(classifier)

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
