import joblib
import torch
import torch.nn as nn
from collections.abc import Iterable
import sys
sys.path.append("../DNN_Architecture")
sys.path.append("../work2_Edgent")
from data_making.utils import layer_FLOPs
from predictors.train import common_func

import function
import MobileNet
import MobileNet2


predictor_dict = {}


def init_predictor_dict(my_dict):
    global predictor_dict
    predictor_dict = my_dict


def kernel_predictor(kernel_kind, edge_device=True):
    """
    根据传入的kernel_kind对DNN层的推理时延进行预测
    input : [['HW','kernel','stride','Cin','Cout','FLOPs']]
    output : ['latency']
    :param kernel_kind : 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param edge_device :True - edge , False - cloud
    :return: conv2d predictors
    """
    de
    if edge_device:
        edge_or_cloud = 0
    else:
        edge_or_cloud = 1

    device_name_list = ["edge","cloud"]
    kernel_name_list = ["conv2d", "dw conv2d", "linear", "maxPool2d", "avgPool2d", "BatchNorm"]
    kernel_path_list = ["../latency_predictor/predictors/config/edge/",
                        "../latency_predictor/predictors/config/cloud/"]
    preditor_name_list = ["_conv2d.pkl", "_dw_conv2d.pkl", "_fc.pkl",
                          "_maxpool2d.pkl", "_avgpool2d.pkl", "_batchNorm.pkl"]

    device_name = device_name_list[edge_or_cloud]
    kernel_path = kernel_path_list[edge_or_cloud]
    kernel_name = kernel_name_list[kernel_kind]
    predictor_name = preditor_name_list[kernel_kind]


    path = kernel_path + mid_str + predictor_name

    # 判断此类型predictor是否已被加载过
    if path in predictor_dict.keys():
        predictor = predictor_dict.get(path)
    else:
        predictor = joblib.load(path)
        predictor_dict[path] = predictor
        # print(f"loading predictors successfully : {kernel_name} - {device_name}")
    return predictor


def predict_latency(features, kernel_kind, edge_device=True,ns=False):
    """
    :param features: input features
    :param kernel_kind: 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    predictor = kernel_predictor(kernel_kind,edge_device,ns=ns)

    predict_lat = predictor.predict(features)
    return predict_lat[0] if predict_lat > 0 else 0


def get_conv2d_lat(input_features,conv2d_layer,edge_device=True,ns=False):
    """
    predict conv2d_layer latency
    conv2d : [['HW','kernel','stride','Cin','Cout','FLOPs']]
    :param input_features: get HW feature
    :param conv2d_layer: conv2d layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    HW = input_features.shape[2]
    kernel = conv2d_layer.kernel_size[0]
    stride = conv2d_layer.stride[0]
    Cin = conv2d_layer.in_channels
    Cout = conv2d_layer.out_channels
    FLOPs = layer_FLOPs.get_conv2d_FLOPs(conv2d_layer,input_features)

    features = [[(HW/stride)*(HW/stride),Cout]] if ns else [[HW,kernel,stride,Cin,Cout,FLOPs]]
    # print(features)
    predict_lat = predict_latency(features,0,edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0



def get_dw_conv2d_lat(input_features, conv2d_layer, edge_device=True,ns=False):
    """
    predict dw conv2d layer latency
    dw_con2d : [['HW','kernel','stride','Cin','FLOPs']]
    :param input_features: get HW feature
    :param conv2d_layer: dw conv2d layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    HW = input_features.shape[2]
    kernel = conv2d_layer.kernel_size[0]
    stride = conv2d_layer.stride[0]
    Cin = conv2d_layer.in_channels
    FLOPs = layer_FLOPs.get_depthwise_separable_conv2d_FLOPs(conv2d_layer,input_features)

    features = [[HW, kernel, stride, Cin, FLOPs]]
    # print(features)
    predict_lat = predict_latency(features, 1, edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0


def get_linear_lat(linear_layer, edge_device=True,ns=False):
    """
    predict linear layer latency
    linear :  [['in_features','out_features','FLOPs']]
    :param linear_layer: linear layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    FLOPs = layer_FLOPs.get_linear_FLOPs(linear_layer)

    features = [[in_features, out_features, FLOPs]]
    predict_lat = predict_latency(features, 2, edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0


def get_maxPool2d_lat(input_features, maxpool2d_layer, edge_device=True,ns=False):
    """
    predict maxPool2d layer latency
    maxPool2d : [['HW','kernel','stride','Cin']]
    :param input_features: get HW feature
    :param maxpool2d_layer: maxPool2d layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    HW = input_features.shape[2]
    Cin = input_features.shape[1]
    kernel = maxpool2d_layer.kernel_size
    stride = maxpool2d_layer.stride

    features = [[HW, kernel, stride, Cin]]
    predict_lat = predict_latency(features, 3, edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0



def get_avgPool2d_lat(input_features, avgpool2d_layer, edge_device=True,ns=False):
    """
    predict avgPool2d layer latency
    avgPool2d : [['HW','output','Cin']]
    :param input_features: get HW feature
    :param avgpool2d_layer: avgPool2d layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    HW = input_features.shape[2]
    output = avgpool2d_layer.output_size[0]
    Cin = input_features.shape[1]

    features = [[HW, output, Cin]]
    # print(features)
    predict_lat = predict_latency(features, 4, edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0


def get_batchNorm_lat(input_features, batchnorm_layer, edge_device=True,ns=False):
    """
    predict batchnorm_layer layer latency
    BatchNorm : [['HW','Cin']]
    :param input_features: get HW feature
    :param batchnorm_layer: batchNorm layer
    :param edge_device :True - edge , False - cloud
    :return: latency
    """
    HW = input_features.shape[2]
    Cin = batchnorm_layer.num_features

    features = [[HW, Cin]]
    predict_lat = predict_latency(features, 5, edge_device,ns=ns)
    return predict_lat if predict_lat > 0 else 0


def predict_kernel_latency(input_features,layer,edge_device=True,ns=False):
    """
    predict kernel latency according to layer's kind
    :param input_features: input x
    :param layer: layer instance
    :param edge_device : True - edge , False - cloud
    :return: latency
    """
    if isinstance(layer,nn.Conv2d):
        if layer.groups == 1:
            return get_conv2d_lat(input_features,layer,edge_device,ns=ns)
        else:
            return get_dw_conv2d_lat(input_features,layer,edge_device,ns=ns)
    elif isinstance(layer,nn.Linear):
        return get_linear_lat(layer,edge_device,ns=ns)
    elif isinstance(layer,nn.MaxPool2d):
        return get_maxPool2d_lat(input_features,layer,edge_device,ns=ns)
    elif isinstance(layer,nn.AdaptiveAvgPool2d):
        return get_avgPool2d_lat(input_features,layer,edge_device,ns=ns)
    elif isinstance(layer,nn.BatchNorm2d):
        return get_batchNorm_lat(input_features,layer,edge_device,ns=ns)
    elif isinstance(layer,nn.Flatten) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.Dropout) or isinstance(layer,nn.ReLU6):
        return 0


def predict_layer_latency(conv_block,x,edge_device,show=False,compare=False,ns=False):
    """
        predict any a layer/block/model latency
        via predict every layer(kernel) latency
        such as :
        alexNet = AlexNet()
        predict_block_latency(alexNet,x,device,epoch,show = False)
        example : ConvNormActivation,InvertedResidual,from GoogLeNet import BasicConv2d
        because they have linked structure, if block has DAG topology,we need design a new function such as `predict_resnet_block_latency`
    """
    if edge_device:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    block_lat = 0.0
    if isinstance(conv_block,Iterable):
        if isinstance(conv_block, ResNet.BasicBlock) or isinstance(conv_block, ResNet2.BasicBlock):  # 一个完整模型中可能会涉及layer = BasicBlock
            block_lat = predict_resnet_block_latency(conv_block, x, edge_device, show=show,ns=ns)
        elif isinstance(conv_block, GoogLeNet.Inception) or isinstance(conv_block, GoogLeNet2.Inception):
            block_lat = predict_inception_block_latency(conv_block, x, edge_device, show=show,ns=ns)
        elif isinstance(conv_block,Inceptionv2.Inception) or isinstance(conv_block,Inceptionv2.Inception):
            block_lat = predict_inception_block_latency(conv_block, x, edge_device, show=show,ns=ns)
        else:
            for layer in conv_block:
                if judge_block(layer):  # 判断是一个自定义块状结构
                    layer_lat = predict_layer_latency(layer, x, edge_device,show=show,ns=ns)
                else:
                    layer_lat = predict_kernel_latency(x,layer,edge_device,ns=ns)
                    if show:
                        if compare:
                            real_lat = common_func.get_block_latency(layer, x, device, epoch=50)
                            print(f"layer : {layer} , real lat : {real_lat:.2f} ms, predict lat : {layer_lat:.2f} ms")
                        else:
                            print(f"layer : {layer} , predict lat : {layer_lat:.2f} ms")
                x = layer(x)
                block_lat += layer_lat
    else:  # not iterable
        block_lat = predict_kernel_latency(x, conv_block, edge_device,ns=ns)
        if show:
            if compare:
                real_lat = common_func.get_block_latency(conv_block, x, device, epoch=50)
                print(f"layer : {conv_block} , real lat : {real_lat:.2f} ms, predict lat : {block_lat:.2f} ms")
            else:
                print(f"layer : {conv_block} , predict lat : {block_lat:.2f} ms")
    return block_lat



def predict_resnet_block_latency(conv_block,x,edge_device,show=False,ns=False):
    """
        predict resnet block latency
        from ResNet import BasicBlock
    """
    features_len = len(conv_block.features)
    down_sample_len = 0 if conv_block.down_sample is None else len(conv_block.down_sample)
    block_lat = 0.0
    origin_x = x

    _index = 0
    for layer in conv_block:
        if _index == features_len + down_sample_len:
            layer_lat = 0.0
        else:
            layer_lat = predict_layer_latency(layer, x, edge_device, show=show,ns=ns)
            x = layer(x)

        block_lat += layer_lat
        _index += 1

        # 重置x让 down_sample 使用
        if _index == features_len:
            x = origin_x

    return block_lat


def predict_inception_block_latency(conv_block,x,edge_device,show = False,ns=False):
    """
        predict inception block latency
        from GoogLeNet import Inception
    """
    len_list = [len(conv_block.branch1),len(conv_block.branch2),len(conv_block.branch3),len(conv_block.branch4)]
    accumulate_len = []
    ac_len = 0
    for i in range(4):
        ac_len += len_list[i]
        accumulate_len.append(ac_len)

    block_lat = 0.0
    origin_x = x

    _index = 0
    for layer in conv_block:
        if _index == accumulate_len[3]:
            layer_lat = 0.0
        else:
            layer_lat = predict_layer_latency(layer, x, edge_device, show=show,ns=ns)
            x = layer(x)

        block_lat += layer_lat
        _index += 1

        if _index == accumulate_len[0] or _index == accumulate_len[1] or _index == accumulate_len[2]:
            x = origin_x

    return block_lat


def skip_layer(layer):
    """
    skip some no important layer
    """
    if isinstance(layer,nn.ReLU) or isinstance(layer,nn.ReLU6) or isinstance(layer,nn.Dropout):
        return True


if __name__ == '__main__':
    init_predictor_dict(my_dict={})

    x = torch.rand((1,3,224,224))
    # x = torch.rand((1, 512, 28, 28))
    model = function.getDnnModel(2)
    # model = nn.Conv2d(512,512,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    model.eval()

    # print(predict_layer_latency(model,x,edge_device=False,show=True))
    for layer in model:
        if skip_layer(layer):
            x = layer(x)
            continue
        # lat = predict_layer_latency(layer,x,edge_device=True,show=True)
        lat = predict_layer_latency(layer,x,edge_device=False,show=True,ns=False)
        x = layer(x)
        print(f"{layer} , latency: {lat:.2f} ms ,  x.shape : {x.shape}")
        print("============================================")






