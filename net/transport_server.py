import pickle

import net_utils
import time
import sys,getopt


"""
    模拟服务端 接收四种不同长度的数据 探究传输时延与数据大小的关系
    server 启动指令 python transport_server.py --epoch=100 -n 1 -s 359 --ip=192.168.3.37
    "-e", "--epoch"         循环epoch次求平均值
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-n", "--network_type"  网络类型 1-3G 2-LTE 3-WIFI
    "-s", "--speed"         带宽数值 
"""
if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:i:p:n:s:", [ "epoch=","ip=","port=","network_type=","speed_bps="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    epoch, ip, port = 500, "192.168.3.37", 8090
    network_type,speed = 3,56
    for opt, arg in opts:
        if opt in ("-e","--epoch"):
            epoch = int(arg)
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-n", "--network_type"):
            network_type = int(arg)
        elif opt in ("-s", "--speed"):
            speed = int(arg)


    p = net_utils.get_socket_server(ip, port)
    totalUseTime = 0.0
    totalUseTime2 = 0.0
    edge_x = None
    for i in range(epoch):
        conn, client = p.accept()

        start_time = time.perf_counter()
        edge_x,ttime = net_utils.get_data(conn)
        end_time = time.perf_counter()

        transport_time = (end_time - start_time) * 1000
        print(f"transport sum time : {transport_time:.3f}ms")
        totalUseTime += transport_time
        totalUseTime2 += ttime
        conn.close()
    # print(f"average time : {(totalUseTime / epoch):.3f}ms")
    print(f"average time : {(totalUseTime2 / epoch):.3f}ms")

    # 展示真实数据以及预测数据
    print("=======================================================")
    speed_Bpms = net_utils.get_speed(network_type,speed)
    data_size_bytes = len(pickle.dumps(edge_x))
    actual_latency = round((totalUseTime2 / epoch),3)
    net_utils.show_speed(data_size_bytes, actual_latency, speed_Bpms)