import torch
import net_utils
import pickle
import time
import sys,getopt



def get_transmission_data(data_type=1):
    """
    获得不同的数据
    :param data_type:
    :return:
    """
    if data_type == 1:
        return torch.rand((1, 3, 224, 224))
    elif data_type == 2:
        x = ""
        for i in range(8):
            x += "aaaaaaaa"
        return x
    elif data_type == 3:
        return torch.rand((1, 10, 224, 224))
    elif data_type == 4:
        return torch.rand((1, 10, 64, 64))


"""
    模拟客户端 发送四种不同长度的数据 探究传输时延与数据大小的关系
    client 启动指令 python  transport_client.py -t 1 -e 100 --ip=192.168.3.37
    "-t", "--type"      获得数据种类
    "-e", "--epoch"     循环epoch次求平均值
    "-i", "--ip"        服务端 ip地址
    "-p", "--port"      服务端 开放端口
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:e:i:p:", ["type=", "epoch=","ip=","port="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    data_type,epoch,ip,port = 1,500,"192.168.3.37",8090
    for opt,arg in opts:
        if opt in ("-t", "--type"):
            data_type = int(arg)
        elif opt in ("-e", "--epoch"):
            epoch = int(arg)
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)

    x = get_transmission_data(data_type=data_type)
    print(len(pickle.dumps(x)))
    totalUseTime = 0.0

    for i in range(epoch):
        conn = net_utils.get_socket_client(ip, port)

        start_time = time.perf_counter()
        net_utils.send_data(conn,x,"data x")
        end_time = time.perf_counter()

        transport_time = (end_time - start_time) * 1000
        # print(f"transport sum time : {transport_time:.3}ms")
        totalUseTime += transport_time
    print(f"average time : {(totalUseTime/epoch):.3f}ms")


