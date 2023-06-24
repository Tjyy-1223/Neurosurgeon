from multiprocessing import Process
import net_utils
import torch
import pickle


def get_bandwidth(conn):
    """
    通过一次信号传输来计算带宽
    :param conn: 连接好的conn
    :return: 带宽 MB/s
    """
    # 得到传输时延
    _,latency = net_utils.get_data(conn)
    # print(f"{latency} ms \n")
    # 计算数据的字节数 Byte 接收数据size固定为[1,3,224,224]
    # data_size = 1 * 3 * 224 * 224 * 8

    # x = torch.rand((1, 3, 224, 224))
    # print(len(pickle.dumps(x)))
    # 得到的数据大小为 602541 bytes
    data_size = 602541

    # 计算带宽 MB/s
    bandwidth = (data_size/1024/1024) / (latency / 1000)
    # print(f"monitor server get bandwidth : {bandwidth} MB/s ")
    return bandwidth


class MonitorServer(Process):
    """
        带宽监视器服务端，其工作流程如下：ip为传入的ip 端口默认为9922
        1. 通过死循环接受带宽监视器客户端传来的数据
        2. 记录传输时间需要的传输时延 (ms)
        3. 计算带宽 并将速度转换成单位 MB/s
        4. 将带宽数据返回给客户端 用于在边缘设备上的模型划分
    """
    def __init__(self,ip,port=9922):
        super(MonitorServer, self).__init__()
        self.ip = ip
        self.port = port

    def run(self) -> None:
        # 创建一个socket服务端
        socket_server = net_utils.get_socket_server(self.ip, self.port)
        # 等待客户端连接 没有客户端连接的话会一直阻塞并等待
        conn, client = socket_server.accept()

        # 获得传输带宽 MB/s
        bandwidth = get_bandwidth(conn)

        # 插入一个break消息接收 防止数据粘包现象
        net_utils.get_short_data(conn)

        # 将获取的带宽传输到客户端
        net_utils.send_short_data(conn,bandwidth,"bandwidth",show=False)

        # 关闭连接
        net_utils.close_conn(conn)
        net_utils.close_socket(socket_server)

