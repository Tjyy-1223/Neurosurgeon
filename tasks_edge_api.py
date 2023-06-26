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
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import random

def start_monitor_client(ip, bandwidth_value):
    """
    开启：带宽监测客户端
    :param ip: 对应的客户端ip地址
    :param bandwidth_value: 需要修改的带宽数据 使用共享变量实现进程通信
    :return:
    """
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
    monitor_cli.start()
    # monitor_cli.join()
    # print(f"bandwidth monitor : get bandwidth value : {bandwidth_value.value} MB/s")


def scheduler_for_bandwidth_monitor_edge(ip, interval, bandwidth_value):
    """
    :param ip: 对应的客户端ip地址
    :param interval: 定时器的定时间隔
    :param bandwidth_value: 需要修改的带宽数据 使用共享变量实现进程通信
    :return:
    """
    # 创建调度器
    scheduler = BackgroundScheduler(timezone='MST')
    # 每隔 interval 秒就创建一个带宽监视进程 用来获取最新带宽
    scheduler.add_job(start_monitor_client, 'interval', seconds=interval, args=[ip,bandwidth_value])
    scheduler.start()



"""
    边缘设备api，用于启动边缘设备，进行前半部分计算后，将中间数据传递给云端设备
    client 启动指令 python tasks_edge_api.py -i 127.0.0.1 -p 9999 -d cpu
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"     是否开启客户端GPU计算 cpu or cuda
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:p:d:", ["ip=","port=","device_on="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    ip,port = "127.0.0.1",999
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")


    # 随机创建任务队列 总共40个DNN推理任务
    tasks_name = ["alex_net", "vgg_net", "le_net", "mobile_net"]
    tasks_list = []
    for i in range(40):
        task_id = random.randint(0, 3)
        tasks_list.append(tasks_name[task_id])
    print(f"tasks list info : {tasks_list}")


    # 使用共享变量 预设定 interval 带宽监测间隔为 1s
    bandwidth_value = multiprocessing.Value('d', 0.0)

    # 在使用调度器持续获取带宽之前 首先获取一次带宽
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
    monitor_cli.start()
    monitor_cli.join()

    interval = 3
    scheduler_for_bandwidth_monitor_edge(ip=ip, interval=interval, bandwidth_value=bandwidth_value)

    print("===================== start inference tasks ===================== ")
    # 准备初始输入数据
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)
    for task_name in tasks_list:
        print(f"get bandwidth value : {bandwidth_value.value} MB/s")
        print(f"get model type: {task_name} ")
        model = get_dnn_model(task_name)

        # 部署阶段 - 选择优化分层点
        upload_bandwidth = bandwidth_value.value  # MBps
        partition_point = neuron_surgeon_deployment(model,network_type="wifi",define_speed=upload_bandwidth,show=False)

        # 使用云边协同的方式开始推理 ： 进行一次连接
        net_utils.start_client(ip,port,x,task_name,partition_point,device)

