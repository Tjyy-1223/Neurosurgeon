import torch
import sys,getopt
from net import net_utils
from utils.inference_utils import get_dnn_model
from deployment import neuron_surgeon_deployment
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from net.monitor_client import MonitorClient
from multiprocessing import Process
import multiprocessing


"""
    边缘设备api，用于启动边缘设备，进行前半部分计算后，将中间数据传递给云端设备
    client 启动指令 python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net
    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "le_net" "mobile_net"
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"     是否开启客户端GPU计算 cpu or cuda
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:i:p:d:", ["type=","ip=","port=","device_on="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    model_type = ""
    ip,port = "127.0.0.1",999
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-t", "--type"):
            model_type = arg
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")


    # 开启：带宽监测客户端
    # 如果没有两个设备测试的条件 可以使用下面的方式 将带宽自定义
    # bandwidth_value = 10  #Mbps
    bandwidth_value = multiprocessing.Value('d', 0.0)
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
    monitor_cli.start()

    # 等待子进程结束后获取到带宽数据
    monitor_cli.join()
    print(f"get bandwidth value : {bandwidth_value.value} MB/s")


    # step2 准备input数据
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)

    # 客户端进行传输
    model = get_dnn_model(model_type)

    # 部署阶段 - 选择优化分层点
    upload_bandwidth = bandwidth_value.value  # MBps
    partition_point = neuron_surgeon_deployment(model,network_type="wifi",define_speed=upload_bandwidth,show=False)

    # 使用云边协同的方式进行模拟
    net_utils.start_client(ip,port,x,model_type,partition_point,device)

