import argparse
import json
import random
import logging
import os

import datasets

from client import *
from server import *

from myThread import MyThread


'''
客户端训练函数，用于多线程时调用
'''
def run(client,global_model,logger):
    diff = client.local_train(global_model,logger)
    return diff


'''
获取控制台参数
'''
def set_args():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--conf', default='./utils/conf.json',type=str,help='训练配置信息')
    parser.add_argument('--log_path', default='./log', type=str, required=False, help='日志保存目录')
    parser.add_argument('--log_file_name', default='main.log', type=str, required=False, help='日志文件名')
    parser.add_argument('--save_model_path', default='./model',type=str, required=False, help='模型保存目录')
    return parser.parse_args()


'''
将日志输出到控制台和日志文件
'''
def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path + '/' + args.log_file_name)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


if __name__ == '__main__':
    # 获取所有的参数
    args = set_args()
    # 创建存放日志和存放模型的文件夹
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 获取日志输出对象
    logger = create_logger(args)

    # 读取配置文件
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    # 获取数据集, 加载描述信息
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    #############################构造NonIID数据################################
    client_idcs = None
    if conf["non_iid"] == 1:
        N_CLIENTS = 10
        DIRICHLET_ALPHA = 1
        N_COMPONENTS = 3
        _, num_cls = train_datasets.data[0].shape[0],  len(train_datasets.classes)
        client_idcs = datasets.mixture_distribution_split_noniid(train_datasets, num_cls, N_CLIENTS, N_COMPONENTS, DIRICHLET_ALPHA)
    ###########################################################################

    # 开启服务器
    server = Server(conf, eval_datasets)
    # 客户端列表
    clients = []

    # 添加10个客户端到列表
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c,client_idcs))

    print("\n\n")

    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样k个进行本轮训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)

        # 权重累计
        weight_accumulator = {}

        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)

        #############################客户端串行训练################################
        if conf["parallel"] == 0:
            # 遍历客户端，每个客户端本地训练模型
            for c in candidates:
                diff = c.local_train(server.global_model,logger)
                
                # 根据客户端的参数差值字典更新总体权重
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])
        ##########################################################################

        #############################客户端并行训练################################
        else:
            
            thread_list = []
            diff_list = []
            # 为每个客户端构造一个本地训练线程
            for c in candidates:
                t = MyThread(run,args=(c,server.global_model,logger))
                thread_list.append(t)
            # 启动每个客户端的训练线程
            for t in thread_list:
                t.start()
            # 当前线程阻塞等待所有客户端完成训练
            for t in thread_list:
                t.join()
            # 获取每个客户端的返回结果
            for t in thread_list:
                diff_list.append(t.get_result())
            # 根据客户端的参数差值字典更新总体权重
            for i in range(int(conf["k"])):
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff_list[i][name])
        ##########################################################################

        # 模型参数聚合
        server.model_aggregate(weight_accumulator,e,args,logger)

        # 模型评估
        acc, loss = server.model_eval()

        logger.info("Epoch {}, acc: {}, loss: {}\n".format(e, acc, loss))
