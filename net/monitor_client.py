import torch
import pickle
from multiprocessing import Process
import net_utils


class MonitorClient(Process):
    """
        带宽监视器客户端，其工作流程如下：
        1. 生成数据并发送给服务端 由服务端记录时间
        2. 获取数据的传输时延 使用进程通信 - 供边缘端进行模型划分
    """
    def __init__(self,ip,port=9922):
        super(MonitorClient, self).__init__()
        self.ip = ip
        self.port = port

    def run(self) -> None:
        # 传入的数据大小
        data = torch.rand((1, 3, 224, 224))

        while True:
            try:
                # 与服务端进行连接 发生意外就一直尝试
                conn = net_utils.get_socket_client(self.ip, self.port)
                # 发送数据
                net_utils.send_data(conn, data, "data", show=False)

                # 插入一个break消息 防止粘包现象
                net_utils.send_short_data(conn,"break",show=False)

                # 直到接收到回应的数据时延 则退出循环
                latency = net_utils.get_short_data(conn)
                print(f"monitor client get latency : {latency} MB/s ")
                if latency is not None:
                    net_utils.close_conn(conn)
                    break
            except ConnectionRefusedError:
                pass
                # print("[Errno 61] Connection refused, try again.")


