
"""
FLOPs参数可以帮助更好地调整预测器 - 这里没有用到
可以进一步用来提升模型预测性能
"""
def get_linear_FLOPs(linear_layer):
    """
    计算全连接层 FLOPs参数
    :param linear_layer:
    :return:
    """
    input_size = linear_layer.in_features
    output_size = linear_layer.out_features
    flops = (2 * input_size - 1) * output_size
    return flops


def get_conv2d_FLOPs(conv2d_layer,x):
    """
    计算conv2d FLOPs参数
    :param conv2d_layer:
    :param x:
    :return:
    """
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]
    padding = conv2d_layer.padding[0]
    stride = conv2d_layer.stride[0]
    input_map = x.shape[2]
    output_map = (input_map - kernel_size + 2 * padding + stride) // stride

    macc = kernel_size * kernel_size * in_channel * out_channel * output_map * output_map
    flops = 2 * macc
    return flops


def get_depthwise_separable_conv2d_FLOPs(conv2d_layer,x):
    """
    计算dw-conv FLOPs参数
    :param conv2d_layer:
    :param x:
    :return:
    """
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size[0] + conv2d_layer.padding[0] + conv2d_layer.stride[0]) / conv2d_layer.stride[0]

    depthwise_macc = kernel_size * kernel_size * in_channel * output_map * output_map
    pointwise_macc = output_map * output_map * in_channel * out_channel
    flops = 2 * (depthwise_macc + pointwise_macc)
    return flops

