import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
from collections.abc import Iterable
import joblib
import os
from models.MobileNet import ConvNormActivation,InvertedResidual
from predictor.get_datasets_func import get_datasets_by_kernel_kind

"""
    1. 主要为构建预测模型 prediction models需要用到的函数
    2. 评估模型准确度用到的函数
    3. 用于预测不同DNN层的kernel_predictor功能函数
"""


def judge_correct(y_real,y_pred,threshold):
    """
    根据阈值范围判断是否预测正确
    :param y_real: 真实值value
    :param y_pred: 预测值value
    :param threshold: 阈值
    :return: True or False
    """
    if (y_pred >= (1 - threshold) * y_real) and (y_pred <= (1 + threshold) * y_real):
        return True
    return False



def get_accuracy(y_real,y_pred,threshold):
    """
    输入python的list : y_real and y_pred
    :param y_real: python list 真实列表
    :param y_pred: python list 预测列表
    :param threshold: 阈值
    :return: 通过阈值threshold判断预测准确率
    """
    res = 0
    if torch.is_tensor(y_pred):
        y_pred = y_pred.numpy().tolist()
        for i in range(len(y_real)):
            if judge_correct(y_real[i], y_pred[i][0], threshold):
                res += 1
    else:
        for i in range(len(y_real)):
            # print(f"real : {y_real[i]} ; pred : {y_pred[i]} ; judge : {judge_correct(y_real[i], y_pred[i], threshold)}")
            if judge_correct(y_real[i], y_pred[i], threshold):
                res += 1

    return round((res/len(y_real)),3)



def model_training_linear(filepath,threshold,get_datasets_func,model_path,save=False):
    """
    从数据集中获取数据，并通过线性回归构建预测模型
    :param filepath:
    :param threshold:
    :param get_datasets_func: 读取数据集的函数
    :param model_path: 模型参数保存位置
    :param save: 是否保存模型参数
    :return: model
    """
    data_features, data_targets = get_datasets_func(filepath)
    x_train, x_test, y_train, y_test = train_test_split(data_features.values, data_targets, test_size=0.1,random_state=0)
    lr = LinearRegression()
    lr.fit(x_train,y_train)

    y_pred_test = lr.predict(x_test)
    print('training model for ' + model_path)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))

    y_pred_train = lr.predict(x_train)
    # acc = get_accuracy(y_train.values.tolist(), y_pred_train, threshold=threshold)
    # print(f"train dataset accuracy : {acc * 100:.2f}%")

    acc = get_accuracy(y_test.values.tolist(), y_pred_test, threshold=threshold)
    print(f"test dataset accuracy : {acc * 100:.2f}%")

    if save:
        joblib.dump(lr,model_path)
    print("train successfully for " + model_path)
    return lr



def load_model(path):
    """
    加载模型并返回模型
    :param path: 模型参数位置
    :return: model
    """
    return joblib.load(path)


def evaluate_model(model,threshold,get_datasets_func):
    """
    评估模型的准确度
    :param model: 已经构建好的模型
    :param threshold: 阈值
    :param get_datasets_func: 获取数据集
    :return:
    """
    data_features, data_targets = get_datasets_func()
    x_train, x_test, y_train, y_test = train_test_split(data_features, data_targets, test_size=0.15, random_state=0)

    y_pred_test = model.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))
    print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

    y_pred_train = model.predict(x_train)
    acc = get_accuracy(y_train.values.tolist(), y_pred_train, threshold=threshold)
    print(f"train dataset accuracy : {acc * 100:.2f}%")

    acc = get_accuracy(y_test.values.tolist(), y_pred_test, threshold=threshold)
    print(f"test dataset accuracy : {acc * 100:.2f}%")



def judge_block(layer):
    """
    判断 layer 是否为一个自定义block类型
    例如 MobileNet、ResNet、googleNet中的block
    """
    if isinstance(layer,ConvNormActivation) or isinstance(layer,InvertedResidual):
        return True
    if isinstance(layer,Iterable):
        return True
    return False


def skip_layer(layer):
    """ 一些不重要的层可以选择跳过"""
    if isinstance(layer,nn.ReLU) or isinstance(layer,nn.ReLU6) or isinstance(layer,nn.Dropout):
        return True



def kernel_predictor_creator(kernel_kind, device, predictor_dict):
    """
    根据传入的kernel_kind对DNN层的推理时延进行预测
    input : [['HW','kernel','stride','Cin','Cout','FLOPs']]
    output : ['latency']
    :param kernel_kind : 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param device : "edge" or "cloud"
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: 指定的预测器 predictor
    """
    current_path = os.path.dirname(__file__) + "/"

    # 存储数据集的位置
    datasets_list = ["conv_lat.csv","dw_conv_lat.csv","linear_lat.csv","maxpool_lat.csv","avgpool_lat.csv","batchnorm_lat.csv"]
    datasets_path = "./dataset/edge/" if device == "edge" else "./dataset/cloud/"

    # 存放模型参数的位置
    predictor_config_list = ["conv.pkl", "dw_conv.pkl", "linear.pkl","maxpool.pkl", "avgpool.pkl", "batchnorm.pkl"]
    predictor_config_path = "./config/edge/" if device == "edge" else "./config/cloud/"

    # predictor根据传入的kernel-kind和device查找有没有对应的模型 如果模型已经被加载 则跳过
    # 如果模型没有被加载 但是有模型参数 则加载模型
    # 否则当场训练模型
    config = current_path + predictor_config_path + predictor_config_list[kernel_kind]

    if config in predictor_dict.keys():
        predictor = predictor_dict.get(config)
    elif os.path.exists(config):
        predictor = joblib.load(config)
        predictor_dict[config] = predictor
        # print("load " + config + " successfully.")
    else:
        datasets = current_path + datasets_path + datasets_list[kernel_kind]
        predictor = model_training_linear(filepath=datasets,
                                          threshold=0.2,
                                          get_datasets_func=get_datasets_by_kernel_kind(kernel_kind),
                                          model_path=config,
                                          save=True)
    # print("get predictor " + config + " successfully.")
    return predictor




def predict_latency(features, kernel_kind, device, predictor_dict):
    """
    加载预测器并预测时延
    :param features: input features
    :param kernel_kind: 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    predictor = kernel_predictor_creator(kernel_kind,device,predictor_dict)

    predict_lat = predictor.predict(features)
    return predict_lat[0] if predict_lat > 0 else 0


def get_conv2d_lat(input_features,conv2d_layer, device, predictor_dict):
    """
    predict conv_layer latency
    conv2d : [['HW','kernel','stride','Cin','Cout','FLOPs']]
    :param input_features: get HW feature
    :param conv2d_layer: conv2d layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    HW = input_features.shape[2]
    kernel = conv2d_layer.kernel_size[0]
    stride = conv2d_layer.stride[0]
    Cin = conv2d_layer.in_channels
    Cout = conv2d_layer.out_channels

    features = [[(HW/stride)*(HW/stride),Cout]]
    predict_lat = predict_latency(features=features,kernel_kind=0,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0



def get_dw_conv2d_lat(input_features, conv2d_layer, device, predictor_dict):
    """
    predict dw conv2d layer latency
    dw_con2d : [['HW','kernel','stride','Cin','FLOPs']]
    :param input_features: get HW feature
    :param conv2d_layer: dw conv2d layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    HW = input_features.shape[2]
    kernel = conv2d_layer.kernel_size[0]
    stride = conv2d_layer.stride[0]
    Cin = conv2d_layer.in_channels

    features = [[HW, kernel, stride, Cin]]
    # print(features)
    predict_lat = predict_latency(features=features,kernel_kind=1,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0


def get_linear_lat(linear_layer, device, predictor_dict):
    """
    predict linear layer latency
    linear :  [['in_features','out_features','FLOPs']]
    :param linear_layer: linear layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features

    features = [[in_features, out_features]]
    predict_lat = predict_latency(features=features,kernel_kind=2,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0


def get_maxPool2d_lat(input_features, maxpool2d_layer, device, predictor_dict):
    """
    predict maxPool2d layer latency
    maxPool2d : [['HW','kernel','stride','Cin']]
    :param input_features: get HW feature
    :param maxpool2d_layer: maxPool2d layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    HW = input_features.shape[2]
    Cin = input_features.shape[1]
    kernel = maxpool2d_layer.kernel_size
    stride = maxpool2d_layer.stride

    features = [[HW, kernel, stride, Cin]]
    predict_lat = predict_latency(features=features,kernel_kind=3,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0



def get_avgPool2d_lat(input_features, avgpool2d_layer, device, predictor_dict):
    """
    predict avgPool2d layer latency
    avgPool2d : [['HW','output','Cin']]
    :param input_features: get HW feature
    :param avgpool2d_layer: avgPool2d layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    HW = input_features.shape[2]
    output = avgpool2d_layer.output_size[0]
    Cin = input_features.shape[1]

    features = [[HW, output, Cin]]
    # print(features)
    predict_lat = predict_latency(features=features,kernel_kind=4,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0


def get_batchNorm_lat(input_features, batchnorm_layer, device, predictor_dict):
    """
    predict batchnorm_layer layer latency
    BatchNorm : [['HW','Cin']]
    :param input_features: get HW feature
    :param batchnorm_layer: batchNorm layer
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    HW = input_features.shape[2]
    Cin = batchnorm_layer.num_features

    features = [[HW, Cin]]
    predict_lat = predict_latency(features=features,kernel_kind=5,device=device,predictor_dict=predictor_dict)
    return predict_lat if predict_lat > 0 else 0



def predict_kernel_latency(input_features,layer, device, predictor_dict):
    """
     predict kernel latency based on the layer's kind
    :param input_features: input x
    :param layer: layer instance
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    if isinstance(layer,nn.Conv2d):
        if layer.groups == 1:
            return get_conv2d_lat(input_features,layer, device, predictor_dict)
        else:
            return get_dw_conv2d_lat(input_features,layer, device, predictor_dict)
    elif isinstance(layer,nn.Linear):
        return get_linear_lat(layer, device, predictor_dict)
    elif isinstance(layer,nn.MaxPool2d):
        return get_maxPool2d_lat(input_features,layer, device, predictor_dict)
    elif isinstance(layer,nn.AdaptiveAvgPool2d):
        return get_avgPool2d_lat(input_features,layer, device, predictor_dict)
    elif isinstance(layer,nn.BatchNorm2d):
        return get_batchNorm_lat(input_features,layer, device, predictor_dict)
    elif isinstance(layer,nn.Flatten) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.Dropout) or isinstance(layer,nn.ReLU6):
        return 0
    else:
        raise RuntimeError("kernel latency 不能预测此种类型的DNN层，请到predictor进行对应的源码修改")




def predict_model_latency(x, model,device, predictor_dict):
    """
    通过预测每一层的时延 来预测一个可循环结构的推理时延
    :param model: 传入的模型
    :param x: 输入数据
    :param device: edge or cloud
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    model_lat = 0.0
    # print(model)
    if isinstance(model,Iterable):
        for layer in model:
            layer_lat = predict_model_latency(x, layer,device, predictor_dict)
            x = layer(x)
            model_lat += layer_lat
    else:  # not iterable
        model_lat = predict_kernel_latency(x, model, device, predictor_dict)
    return model_lat
