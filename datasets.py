import torch 
from torchvision import datasets, transforms
import numpy as np 
import random

'''
获取数据集
'''
def get_dataset(dir, name):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	
	return train_dataset, eval_dataset

'''
功能函数
'''
def avg_divide(l, g):
    """
    将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
    每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
    返回由不同的groups组成的列表
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

'''
功能函数
'''
def split_list_by_idcs(l, idcs):
    """
    将列表`l` 划分为长度为 `len(idcs)` 的子列表
    第`i`个子列表从下标 `idcs[i]` 到下标`idcs[i+1]`
    （从下标0到下标`idcs[0]`的子列表另算）
    返回一个由多个子列表组成的列表
    """
    res = []
    current_index = 0
    for index in idcs: 
        res.append(l[current_index: index])
        current_index = index
    return res

'''
实现NonIID数据集的构造
'''
def mixture_distribution_split_noniid(dataset, n_classes, n_clients, n_clusters, alpha):
    if n_clusters == -1:
        n_clusters = n_classes
    all_labels = list(range(n_classes))
    np.random.shuffle(all_labels)
    
    clusters_labels = avg_divide(all_labels, n_clusters)
    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    data_idcs = list(range(len(dataset)))
    # 记录每个cluster大小的向量
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    # 存储每个cluster对应的数据索引
    clusters = {k: [] for k in range(n_clusters)}
    for idx in data_idcs:
        _, label = dataset[idx]
        # 由样本数据的label先找到其cluster的id
        group_id = label2cluster[label]
        # 再将对应cluster的大小+1
        clusters_sizes[group_id] += 1
        # 将样本索引加入其cluster对应的列表中
        clusters[group_id].append(idx)

    # 将每个cluster对应的样本索引列表打乱
    for _, cluster in clusters.items():
        random.shuffle(cluster)
    # 记录某个cluster的样本分到某个client上的数量
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64) 

    # 遍历每一个cluster
    for cluster_id in range(n_clusters):
        # 对每个client赋予一个满足dirichlet分布的权重，用于该cluster样本的分配
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        # np.random.multinomial 表示投掷骰子clusters_sizes[cluster_id](该cluster中的样本数)次，落在各client上的权重依次是weights
        # 该函数返回落在各client上各多少次，也就对应着各client应该分得来自该cluster的样本数
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    # 对每一个cluster上的每一个client的计数次数进行前缀（累加）求和，
    # 相当于最终返回的是每一个cluster中按照client进行划分的样本分界点下标
    clients_counts = np.cumsum(clients_counts, axis=1)
    
    clients_idcs = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        # cluster_split为一个cluster中按照client划分好的样本
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        # 将每一个client的样本累加上去
        for client_id, idcs in enumerate(cluster_split):
            clients_idcs[client_id] += idcs
    return clients_idcs

# 获取NonIID数据的测试代码
# import matplotlib.pyplot as plt
# torch.manual_seed(42)

# if __name__ == "__main__":

#     N_CLIENTS = 10
#     DIRICHLET_ALPHA = 1
#     N_COMPONENTS = 3

#     #, split="byclass"
#     train_data = datasets.CIFAR10(root=".", download=True, train=True)
#     test_data = datasets.CIFAR10(root=".", download=True, train=False)
#     n_channels = 1


#     input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes)


#     train_labels = np.array(train_data.targets)

#     # 注意每个client不同label的样本数量不同，以此做到Non-IID划分
#     client_idcs = mixture_distribution_split_noniid(train_data, num_cls, N_CLIENTS, N_COMPONENTS, DIRICHLET_ALPHA)
#     print(client_idcs)

#     # 展示不同client的不同label的数据分布
#     plt.figure(figsize=(20,3))
#     plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
#             bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
#             label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
#     plt.xticks(np.arange(num_cls), train_data.classes)
#     plt.legend()
#     plt.show()
