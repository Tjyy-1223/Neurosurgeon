import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from collections import abc, OrderedDict

""" conv + norm + activation """
class ConvNormActivation(torch.nn.Sequential):
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
        layers = [torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation=dilation,groups=groups,bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels




class InvertedResidual_SentenceIterator(abc.Iterator):
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


""" conv：pw + dw + pw """
class InvertedResidual(nn.Module):
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
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
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



def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SentenceIterator(abc.Iterator):
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


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            round_nearest: int = 8,
            dropout: float = 0.2,
    ) -> None:
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            # expand ratio,out channel,range epoch,stride
            [1, 16, 1, 1],
            [1, 24, 2, 2],
            [0.5, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 1, 2],
            [6, 160, 2, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features: List[nn.Module] = [
            ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        # building last several layers
        features.append(
            ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,activation_layer=nn.ReLU6)
        )

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
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


def mobilenet_v2(**kwargs: Any) -> MobileNetV2:
    model = MobileNetV2(**kwargs)
    return model
