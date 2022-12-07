# 基于python实现横向联邦图像分类

## 简介

基于pytorch框架，在一台计算机上模拟联邦学习场景，实现横向联邦图像分类任务。基于FederatedAI开源教程进行二次开发，利用多线程实现了客户端并行训练，并实现了NonIID数据场景。



## 运行代码

```shell
python main.py --conf ./utils/conf.json --log_path ./log --log_file_name main.log --save_model_path ./model
```



## 主要功能

- 训练模式：一个中心参数服务器节点保存一个全局模型，n个客户端节点使用各个节点本地数据训练一个本地模型，训练流程可分为如下四个步骤。

  1. n个客户端节点分别独立的在本地数据上训练一个本地模型。
  2. n个客户端节点将本地模型参数上传到服务器节点。
  3. 服务器节点对搜集到的客户端模型参数进行聚合。
  4. 服务器节点将模型下发到各个客户端节点，循环上述过程。

- 支持多线程：通过设置conf.json文件中的parallel参数可以实现客户端节点并行训练。

- 支持NonIID数据采样：通过设置conf.json文件中的non_iid参数可以使得各个节点本地数据之间非独立同分布。

  

## 文件介绍

- client.py 客户端节点实现
- server.py 服务器节点实现
- datasets.py 数据集读取模块，实现NonIID数据集的构造
- models.py 模型读取模块
- myThread.py 线程模块
- main.py 用于训练模型
- utils 存放conf.json配置文件
- data 存放数据集
- log 存放训练日志
- model 存放训练模型
- (data、log、model文件夹会自动创建）



## 配置信息

- model_name：模型名称
- no_models：客户端数量
- type：数据集信息
- global_epochs：全局迭代次数，即服务端与客户端的通信迭代次数
- local_epochs：本地模型训练迭代次数
- k：每一轮迭代时，服务端会从所有客户端中挑选k个客户端参与训练。
- batch_size：本地训练每一轮的样本数
- lr，momentum，lambda：本地训练的超参数设置
- parallel：1表示开启多线程，客户端节点并行训练，0则关闭
- non_iid： 1表示各个节点数据非独立同分布，0则独立同分布



  ## 参考资料

- [联邦学习教程](https://github.com/FederatedAI/Practicing-Federated-Learning)
- [NonIID数据构造教程](https://www.cnblogs.com/orion-orion/p/15991423.html)