from monitor_server import MonitorServer
from monitor_client import MonitorClient

if __name__ == '__main__':
    ip = "127.0.0.1"

    monitor_ser = MonitorServer(ip=ip)
    monitor_cli = MonitorClient(ip=ip)

    monitor_ser.start()
    monitor_cli.start()

    monitor_cli.join()
    monitor_ser.join()
