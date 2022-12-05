import threading
import time

'''
对线程进行封装，可以读取函数的返回值
'''
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self) # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

# 测试代码
# def run():
#     time.sleep(2)
#     print('当前线程的名字是： ', threading.current_thread().name)
#     time.sleep(2)
#     return threading.current_thread().name

# if __name__ == '__main__':
#     thread_list = []
#     for i in range(5):
#         t = MyThread(run)
#         thread_list.append(t)

#     for t in thread_list:
#         t.start()

#     for t in thread_list:
#         t.join()

#     for t in thread_list:
#         print(t.get_result())
