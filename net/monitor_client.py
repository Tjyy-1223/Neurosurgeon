import time

import torch
import pickle
from multiprocessing import Process
from net import net_utils
from apscheduler.schedulers.blocking import BlockingScheduler
import multiprocessing


class MonitorClient(Process):
    """
        带宽监视器客户端，其工作流程如下：通过定时机制每隔一段时间测量一次
        1. 生成数据并发送给服务端 由服务端记录时间
        2. 获取数据的传输时延 使用进程通信 - 供边缘端进行模型划分
    """
    def __init__(self, ip, bandwidth_value, port=9922, interval=3):
        super(MonitorClient, self).__init__()
        self.ip = ip
        self.bandwidth_value = bandwidth_value
        self.port = port
        self.interval = interval


    def start_client(self) -> None:
        # 传入的数据大小
        data = torch.rand((1, 3, 224, 224))

        while True:
            try:
                # 与服务端进行连接 发生意外就一直尝试
                conn = net_utils.get_socket_client(self.ip, self.port)
                # 发送数据
                net_utils.send_data(conn, data, "data", show=False)

                # 插入一个break消息 防止粘包现象
                net_utils.send_short_data(conn, "break", show=False)

                # 直到接收到回应的数据时延 则退出循环
                latency = net_utils.get_short_data(conn)
                # print(f"monitor client get latency : {latency} MB/s ")
                if latency is not None:
                    self.bandwidth_value.value = latency
                    net_utils.close_conn(conn)
                    break
                time.sleep(1)
            except ConnectionRefusedError:
                pass
                # print("[Errno 61] Connection refused, try again.")

    def schedular(self):
        # 使用定时机制 每隔一段时间后监测带宽
        # 创建调度器
        scheduler = BlockingScheduler()

        # 添加任务
        scheduler.add_job(self.start_client, 'interval', seconds=self.interval)
        scheduler.start()


    def run(self) -> None:
        # self.schedular()
        self.start_client()


if __name__ == '__main__':
    ip = "127.0.0.1"
    bandwidth_value = multiprocessing.Value('d', 0.0)
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)

    monitor_cli.start()
    monitor_cli.join()