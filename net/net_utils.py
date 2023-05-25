import socket
import time
import pickle
import torch
import platform

def get_socket_server(ip, port, max_client_num=10):
    """
    为服务端 - 云端设备创建一个socket 用来等待客户端连接
    :param ip: 云端设备机器的ip
    :param port: socket的网络端口
    :param max_client_num: 最大可连接的用户数
    :return: 创建好的socket
    """
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket

    # 判断使用的是什么平台
    sys_platform = platform.platform().lower()
    if "windows" in sys_platform:
        p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # windows
    else:
        p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) # macos or linux

    p.bind((ip, port))  # 绑定端口号
    p.listen(max_client_num)  # 打开监听
    return p


def get_socket_client(ip, port):
    """
    客户端(边端设备)创建一个socket 用于连接云端设备
    :param ip: 要连接的云端设备机器的ip
    :param port: 云端设备socket的端口
    :return: 创建好的连接
    """
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((ip, port))
    return conn


def close_conn(conn):
    """
    边端设备 终止conn连接
    :param conn: conn连接
    :return: 终止连接
    """
    conn.close()



def close_socket(p):
    """
    云端设备 关闭socket
    :param p: socket
    :return:关闭连接
    """
    p.close()


def wait_client(p):
    """
    等待一次conn连接
    :param p: socket
    :return:
    """
    conn, client = p.accept()
    print(f"successfully connection :{conn}")
    return conn,client



def create_server(p):
    """
    使用socket 建立一个 server - 循环等待客户端发来请求
    :param p: socket连接
    :return: None
    """
    while True:
        conn, client = p.accept()  # 接收到客户端的请求
        print(f"connect with client :{conn} successfully ")

        sum_time = 0.0
        # 收发消息
        data = [conn.recv(1)]  # 为了更准确地记录时间，先获取长度为1的消息，之后开启计时
        while True:
            start_time = time.perf_counter()  # 记录开始时间
            packet = conn.recv(1024)
            end_time = time.perf_counter()  # 记录结束时间
            transport_time = (end_time - start_time) * 1000
            sum_time += transport_time  # 传输时间累计到sum_time变量中

            data.append(packet)
            if len(packet) < 1024:  # 长度 < 1024 代表所有数据已经被接受
                break

        parse_data = pickle.loads(b"".join(data))  # 发送和接收数据都使用pickle包，所以这里进行解析pickle
        print(f"get all data come from :{conn} successfully ")

        if torch.is_tensor(parse_data):  # 主要对tensor数据进行数据大小的衡量
            total_num = 1
            for num in parse_data.shape:
                total_num += num
            data_size = total_num * 4
        else:
            data_size = 0.0

        print(f"data size(bytes) : {data_size} \t transfer time : {sum_time:.3} ms")
        print("=====================================")
        conn.send("yes".encode("UTF-8"))  # 接收到所有请求后回复client
        conn.close()



def send_data(conn, x, msg="msg"):
    """
    向另一方发送较长数据 例如DNN模型中间层产生的tensor
    注意：接收数据需要使用get_data函数
    这个send_data消息主要分为： 发送数据长度 - 接收回应 - 发送真实数据 - 接收回应
    :param conn: 客户端的conn连接
    :param x: 要发送的数据
    :param msg: 对应的 提示
    :return:
    """
    send_x = pickle.dumps(x)
    conn.sendall(pickle.dumps(len(send_x)))
    resp_len = conn.recv(1024).decode()

    conn.sendall(send_x)
    resp_data = conn.recv(1024).decode()
    print(f"get {resp_data} , {msg} has been sent successfully")  # 表示对面已收到数据



def send_short_data(conn,x,msg="msg"):
    """ 向另一方发送比较短的数据 接收数据直接使用get_short_data"""
    send_x = pickle.dumps(x)
    conn.sendall(send_x)
    print(f"short message , {msg} has been sent successfully")  # 表示对面已收到数据



def get_data(conn):
    """
    获取一次长数据 主要分为 获取数据长度 - 回应 - 获取数据 - 回应
    :param conn: 建立好的连接
    :return: 解析后的数据 和 获取数据消耗的时延
    """
    # 接收数据长度
    data_len = pickle.loads(conn.recv(1024))
    conn.sendall("yes len".encode())

    # 接收数据并记录时延
    sum_time = 0.0
    data = [conn.recv(1)]
    while True:
        start_time = time.perf_counter()
        packet = conn.recv(40960)
        end_time = time.perf_counter()
        transport_time = (end_time - start_time) * 1000
        sum_time += transport_time

        data.append(packet)
        if len(b"".join(data)) >= data_len:
            break
        # if len(packet) < 4096: break

    parse_data = pickle.loads(b"".join(data))
    conn.sendall("yes".encode())
    return parse_data,sum_time


def get_short_data(conn):
    """ 获取短数据"""
    return pickle.loads(conn.recv(1024))