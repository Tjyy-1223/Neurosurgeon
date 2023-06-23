from multiprocessing import Process
import os

from multiprocessing import  Process

class MyProcess(Process): # 继承Process类
    def __init__(self,name):
        super(MyProcess,self).__init__()
        self.name = name

    def run(self):
        print('测试%s多进程' % self.name)


if __name__ == '__main__':
    process_list = []
    for i in range(5):  # 开启5个子进程执行fun1函数
        p = MyProcess('Python') # 实例化进程对象
        p.start()
        process_list.append(p)

    # for i in process_list:
        # p.join()

    print('结束测试')

# 子进程要执行的代码
# def run_proc(name):
#     print('Run child process %s (%s)...' % (name, os.getpid()))
#
# if __name__=='__main__':
#     print('Parent process %s.' % os.getpid())
#     p = Process(target=run_proc, args=('test',))
#     print('Child process will start.')
#     p.start()
#     p.join()
#     print('Child process end.')