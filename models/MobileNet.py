import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable,List, Optional
from collections import abc

class MobileNet(nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            round_nearest: int = 8,
            dropout: float = 0.2,
    ) -> None:
        """
            input_channels: 输入图像的通道数，默认通道数为3
            num_classes: MobileNetV2的输出维度，默认为1000
        """
        super(MobileNet, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        input_channels_modify = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            # expand ratio,out channel,range epoch,stride
            [1, 16, 1, 1],
            [1, 24, 2, 2],
            [0.5, 64, 4, 2],
            [0.5, 96, 3, 1],
            [6, 160, 1, 2],
            [6, 160, 2, 2],
            [6, 320, 1, 1],
        ]

        # 构建模型第一层
        input_channels_modify = _make_divisible(input_channels_modify * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features: List[nn.Module] = [
            ConvNormActivation(input_channels, input_channels_modify, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]

        # 构建 inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channels_modify, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channels_modify = output_channel

        # 构建模型最后一层
        features.append(
            ConvNormActivation(input_channels_modify, self.last_channel, kernel_size=1, norm_layer=norm_layer,activation_layer=nn.ReLU6)
        )


        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )
        self.len1 = len(self.features)
        self.len2 = len(self.classifier)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def __len__(self):
        return len(self.features) + len(self.classifier)

    def __iter__(self):
        """ 用于遍历MobileNetV2模型的每一层 """
        return SentenceIterator(self.features,self.classifier)

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
        MobileNetV2迭代器
        下面是 MobileNetV2 网络的迭代参数调整
        将下面的设置传入到 MobileNetV2 的 __iter__ 中可以完成对于 MobileNetV2 网络的层级遍历
    """
    def __init__(self,features, classifier):
        self.features = features
        self.classifier = classifier

        self._index = 0

        self.len1 = len(features)
        self.len2 = len(classifier)

    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]
            else:
                len = self.len1
                layer = self.classifier[self._index - len]

        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer


class ConvNormActivation(torch.nn.Sequential):
    """
        MobileNet-v2 ConvNormActivation-DNN块设计
        由三个部分组成：Conv + Norm + Activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(kernel_size,kernel_size),
                            stride=(stride,stride),
                            padding=padding,
                            dilation=(dilation,dilation),
                            groups=groups,
                            bias=bias)
        ]
        if norm_layer is not None:
            layers.append(
                norm_layer(out_channels)
            )
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(
                activation_layer(**params)
            )
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels


class InvertedResidual(nn.Module):
    """
        MobileNet-v2 ConvNormActivation-InvertedResidualBlock设计
        由下面几个部分组成：
        1) ConvNormActivation block
        2) ConvNormActivation block
        3) Conv2d layer
        4) Norm layer
    """
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        layers.extend([
                # dw
                ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,activation_layer=nn.ReLU6),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=(1,1), stride=(1,1), padding=0, bias=False),
                norm_layer(oup),
            ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1


    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def __iter__(self):
        return InvertedResidual_SentenceIterator(self.conv)
    def __len__(self):
        return len(self.conv)
    def __getitem__(self, item):
        try:
            if item < len(self.conv):
                layer = self.conv[item]
            else:
                raise StopIteration()
        except IndexError:
            raise StopIteration()
        return layer



class InvertedResidual_SentenceIterator(abc.Iterator):
    """
        InvertedResidual迭代器
        下面是 InvertedResidual 网络的迭代参数调整
        将下面的设置传入到 InvertedResidual 的 __iter__ 中可以完成对于 InvertedResidual 网络的层级遍历
    """
    def __init__(self,conv):
        self.conv = conv
        self._index = 0
        self.len = len(conv)

    def __next__(self):
        try:
            layer = self.conv[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer



def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
