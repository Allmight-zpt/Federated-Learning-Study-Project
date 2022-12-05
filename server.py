import models
import torch

'''
服务端对象的实现
'''
class Server(object):
	'''
	定义构造函数
	'''
	def __init__(self, conf, eval_dataset):
		# 导入配置文件
		self.conf = conf 
		# 根据配置获取模型文件
		self.global_model = models.get_model(self.conf["model_name"]) 
		# 生成一个测试集合加载器
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	'''
	定义客户端模型聚合函数，即全局模型的更新函数
	weight_accumulator 存储了每一个客户端的上传参数的变化值
	'''
	def model_aggregate(self, weight_accumulator):
		# 遍历全局模型的每一层
		for name, data in self.global_model.state_dict().items():
			# 计算每一层的更新值
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			# 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
	
	'''
	定义评估函数
	'''
	def model_eval(self):
		# 开启模型评估模式（不修改参数）
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		# 遍历评估数据集合
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			# 获取所有的样本总量大小
			dataset_size += data.size()[0]
			# 加载到gpu
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			# 加载到模型中训练
			output = self.global_model(data)
			# 聚合所有的损失 cross_entropy交叉熵函数计算损失
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item()
			# 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
			pred = output.data.max(1)[1]
			# 统计预测结果与真实标签target的匹配总个数
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
		# 准确率
		acc = 100.0 * (float(correct) / float(dataset_size))
		# 损失值
		total_l = total_loss / dataset_size
		return acc, total_l