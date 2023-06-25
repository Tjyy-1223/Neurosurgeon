import pickle
import torch
from utils import inference_utils
from predictor import predictor_utils
from net import net_utils

def get_layer(model,point):
    """
    get model's partition layer
    """
    if point == 0:
        layer = None
    else:
        layer = model[point - 1]
    return layer


def get_input(HW):
    """
    根据HW生成相应的pytorch数据 -> torch(1,3,224,224)
    :param HW: HW表示输入的高度和宽度
    :return: torch数据
    """
    return torch.rand(size=(1, 3, HW, HW), requires_grad=False)


def neuron_surgeon_deployment(model,network_type,define_speed,show=False):
    """
    为DNN模型选取最优划分点
    :param model: DNN模型
    :param network_type: 3g or lte or wifi
    :param define_speed: bandwidth
    :param show: 是否展示
    :return: 选取的最优partition_point
    """
    res_lat = None
    res_index = None
    res_layer_index = None
    predictor_dict = {}

    layer_index = 0     # 标记layer顺序 - 指no-skip layer
    for index in range(len(model) + 1):
        if index != 0 and predictor_utils.skip_layer(model[index - 1]):
            continue

        x = get_input(HW=224)
        #   index = 0 : edge_model is None
        #   index = len(model) : cloud model is None
        #   partition the model
        edge_model,cloud_model = inference_utils.model_partition(model,index)

        #   predict edge latency
        edge_lat = predictor_utils.predict_model_latency(x,edge_model,device="edge",predictor_dict=predictor_dict)
        x = edge_model(x)

        #   predict transmission latency,network_type = WI-FI
        transport_size = len(pickle.dumps(x))
        speed = net_utils.get_speed(network_type=network_type,bandwidth=define_speed)
        transmission_lat = transport_size / speed

        # if index == len(model):
        #     transmission_lat = 0.0

        #   predict cloud latency
        cloud_lat = predictor_utils.predict_model_latency(x, cloud_model,device="cloud",predictor_dict=predictor_dict)

        # show detail
        total_lat = edge_lat + transmission_lat + cloud_lat

        now_layer = get_layer(model,index)
        if show:
            print(f"index {layer_index + 1} - layer : {now_layer} \n"
                  f"edge latency : {edge_lat:.2f} ms , transmit latency : {transmission_lat:.2f} ms , "
                  f"cloud latency : {cloud_lat:.2f} ms , total latency : {total_lat:.2f} ms")
            print(
                "----------------------------------------------------------------------------------------------------------")

        #   get best partition point
        #   index - real layer index : get layer
        #   layer_index : only used as flag
        if res_lat is None or total_lat < res_lat:
            res_lat = total_lat
            res_index = index
            res_layer_index = layer_index
        layer_index += 1


    # show best partition point
    res_layer = get_layer(model,res_index)
    print(f"best latency : {res_lat:.2f} ms , best partition point : {res_layer_index} - {res_layer}")
    print("----------------------------------------------------------------------------------------------------------")

    #   return the best partition point
    return res_index




