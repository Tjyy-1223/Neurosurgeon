import torch
import torch.nn as nn
from collections import abc

class AlexNet(nn.Module):
    def __init__(self, input_channels=3, num_classes: int = 1000) -> None:
        """
        input_channels: 输入图像的通道数，默认通道数为3
        num_classes: AlexNet的输出维度，默认为1000
        """
        super(AlexNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels,64,kernel_size=(11,11),stride=(4,4),padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.len = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def __iter__(self):
        """ 用于遍历AlexNet模型的每一层 """
        return SentenceIterator(self.layers)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        layer = nn.Sequential()
        try:
            if index < self.len:
                layer = self.layers[index]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    """
    AlexNet迭代器
    下面是 AlexNet 网络的迭代参数调整
    将下面的设置传入到 AlexNet 的 __iter__ 中可以完成对于 AlexNet 网络的层级遍历
    """
    def __init__(self, layers):
        self.layers = layers
        self._index = 0
        self.len = len(layers)

    def __next__(self):
        layer = nn.Sequential()
        try:
            if self._index <= self.len:
                layer = self.layers[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer
