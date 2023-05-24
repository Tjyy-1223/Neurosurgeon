import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
from collections.abc import Iterable
import joblib

from models.MobileNet import ConvNormActivation,InvertedResidual


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
    :return: None
    """
    data_features, data_targets = get_datasets_func(filepath)
    x_train, x_test, y_train, y_test = train_test_split(data_features, data_targets, test_size=0.1,random_state=0)
    lr = LinearRegression()
    lr.fit(x_train,y_train)

    y_pred_test = lr.predict(x_test)
    if save:
        joblib.dump(lr,model_path)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))
    print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

    y_pred_train = lr.predict(x_train)
    acc = get_accuracy(y_train.values.tolist(), y_pred_train, threshold=threshold)
    print(f"train dataset accuracy : {acc * 100:.2f}%")

    acc = get_accuracy(y_test.values.tolist(), y_pred_test, threshold=threshold)
    print(f"test dataset accuracy : {acc * 100:.2f}%")

    # 用交叉验证计算得分
    # score_pre = cross_val_score(lr, data_features, data_targets, cv=10).mean()
    # print(score_pre)



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


