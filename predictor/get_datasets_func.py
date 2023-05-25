import pandas as pd


def get_datasets_by_kernel_kind(kernel_kind):
    """
    根据传入的kernel_kind - 返回对应的处理函数
    :param kernel_kind: kernel_kind : 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :return: features,targets
    """
    if kernel_kind == 0:
        return get_datasets_for_conv
    elif kernel_kind == 1:
        return get_datasets_for_dw_conv
    elif kernel_kind == 2:
        return get_datasets_for_linear
    elif kernel_kind == 3:
        return get_datasets_for_maxpool
    elif kernel_kind == 4:
        return get_datasets_for_avgpool
    elif kernel_kind == 5:
        return get_datasets_for_batchnorm
    else:
        raise RuntimeError("没有这种类型的DNN层")



def get_datasets_for_conv(filepath):
    """
    对于卷积层 conv
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)

    data_in = data[['HW','kernel','stride','Cin','Cout','FLOPs']]
    data_in.loc[:, 'new_kernel'] = (data_in['HW']/data_in['stride']) * (data_in['HW']/data_in['stride'])
    data_features = data_in[['new_kernel','Cout']]
    data_targets = data['latency']
    return data_features,data_targets



def get_datasets_for_dw_conv(filepath):
    """
    对于深度可分离卷积层 dw-conv
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)

    data_in = data[['HW','kernel','stride','Cin','FLOPs']]
    data_in.loc[:, 'new_kernel'] = (data_in['HW']/data_in['stride']) * (data_in['HW']/data_in['stride'])
    data_features = data_in[['HW','kernel','stride','Cin']]

    # print(data_features[:10])
    # data_features = data[['FLOPs']]
    data_targets = data['latency']

    return data_features,data_targets


def get_datasets_for_linear(filepath):
    """
    对于全连接层 linear
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)
    data_in = data[['in_features','out_features']]
    data_targets = data['latency']

    return data_in,data_targets


def get_datasets_for_maxpool(filepath):
    """
    对于最大池化层 maxpool
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)

    data_in = data[['HW','kernel','stride','Cin']]
    data_targets = data['latency']

    return data_in,data_targets


def get_datasets_for_avgpool(filepath):
    """
    对于平均池化层 avgpool
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)

    data_in = data[['HW', 'output', 'Cin']]
    data_targets = data['latency']

    return data_in, data_targets



def get_datasets_for_batchnorm(filepath):
    """
    对于batchnorm层
    从数据集中获取 features 和 target
    :param filepath: 数据集存在的位置
    :return: 构建model需要的属性和特征
    """
    # 从csv文件中读取数据
    data = pd.read_csv(filepath)

    data_in = data[['HW','Cin']]
    data_targets = data['latency']

    return data_in,data_targets