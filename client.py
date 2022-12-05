import models
import torch
import copy

'''
客户端对象的实现
'''
class Client(object):
	'''
	定义构造函数
	'''
	def __init__(self, conf, model, train_dataset, id = -1,client_idcs = None):
		# 导入配置文件
		self.conf = conf
		# 根据配置文件获取模型
		self.local_model = models.get_model(self.conf["model_name"]) 
		# 定义客户端id
		self.client_id = id
		# 定义数据集
		self.train_dataset = train_dataset
		#############################IID数据################################
		if client_idcs == None:
			print("IID")
			# 按客户端id获取数据
			# 因为server.py中shuffle = True因此此处得到的时IID数据
			all_range = list(range(len(self.train_dataset)))
			data_len = int(len(self.train_dataset) / self.conf['no_models'])
			train_indices = all_range[id * data_len: (id + 1) * data_len]
			# 定义数据加载器
			self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
										sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
		####################################################################

		############################NonIID数据##############################
		else:
			print("NonIID")
			# 定义数据加载器
			self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
										sampler=torch.utils.data.sampler.SubsetRandomSampler(client_idcs[self.client_id]))
		####################################################################
		
		
									
	'''
	定义客户端本地训练函数
	'''
	def local_train(self, model):
		# 客户端首先用服务器端下发的全局模型覆盖本地模型
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	 
		# 定义最优化函数器用于本地模型训练
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])

		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				# 加载到gpu
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				# 梯度
				optimizer.zero_grad()
				# 预测
				output = self.local_model(data)
				# 计算损失函数 cross_entropy 交叉熵误差
				loss = torch.nn.functional.cross_entropy(output, target)
				# 反向传播
				loss.backward()
				# 更新参数
				optimizer.step()
			print("Epoch %d done." % e)	
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			# 计算训练前后的差值，最终会返回到服务端更新服务端的模型
			diff[name] = (data - model.state_dict()[name])
		return diff
		