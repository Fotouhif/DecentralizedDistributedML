import random, math, sys
import torch
from torch.utils.data import DataLoader#, Sampler, DistributedSampler
import numpy as np
import pandas as pd
import random

class Partition(object):
	
	def __init__(self, data, index):
		self.data = data
		self.index = index
		
	def __len__(self):
		return len(self.index)
	
	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]
	
	
class DataPartition(object):
	
	def __init__(self, dataset, num_cam=30, **kwargs):
		self.data = dataset
		self.num_cam = num_cam
		self.num_workers = kwargs.get('num_workers')
		self.dataset_name = kwargs.get('data')
		self.data_dist = kwargs.get('data_dist', 'iid')
		self.dist = kwargs['dist']
		#print(type(self.data.targets))

		#split inputs and labels
		if self.dataset_name == 'ImageNet':
			self.labels = self.data.targets
		else:
			self.labels = [self.data[i][1] for i in range(len(self.data))]
		#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		#print([self.data[i][1] for i in range(len(self.data))])
		#self.labels = self.data.targets # got error for no attributes "targets"
		if self.dataset_name == 'MNIST':
			self.labels = np.array(self.labels)
			self.labels = np.asarray(self.labels)
		elif self.dataset_name in ['CIFAR10', 'CIFAR100', 'stl10', 'Agdata', 'Agdata-small', 'ImageNet']:
			self.labels = np.asarray(self.labels)
			# print('label', len(self.labels))
		# print(self.data.classes)
		# if self.dataset_name == 'Agdata':
		# 	classes = [0]
		# 	class_counts = len(self.data)
		# 	self.num_classes = len(classes)
		# 	self.class_list = self.data
		# 	print(self.class_list)
		# else:
		#identify number of classes
		classes, class_counts = np.unique(self.labels, return_counts=True)
		# print(classes)
		self.num_classes = len(classes)
		# print('numclass',self.num_classes)
		min_cls_cnt = min(class_counts) # used in non-iid cases to ensure equal data distribution
		#sort data by class
		self.class_list = [[] for nb in range(self.num_classes)]
		for class_ in range(self.num_classes):
			self.class_list[class_] = np.argwhere(self.labels == class_).ravel()
		# print('class_list', self.class_list)

		#initialize worker index
		self.worker_index = [[] for worker in range(self.num_workers)]
		# print(self.num_classes)
		# print(self.num_workers)

		#Partition into desired distribution
		#iid
		if self.data_dist == 'iid':
			#print('here')
			#distribute data amongst workers (card dealer style)
			sample_per_class = [len(class_)//self.num_workers for class_ in self.class_list] # To account for imbalanced classes like MNIST
			#print("sample_per_class =", sample_per_class)
			start_id = [0 for _ in range(self.num_classes)] # start ID for each class -- Same reason as above

			#iterate through each worker
			for rank_ in range(self.num_workers):
				# iterate through each class
				for class_ in range(self.num_classes):
					temp_index = self.class_list[class_][start_id[class_]:start_id[class_]+sample_per_class[class_]] # Extract data from class
					#print("temp_index = ", temp_index)
					self.worker_index[rank_].extend(temp_index) # Assign data to worker
					start_id[class_] += sample_per_class[class_] # Updates start ID for each class

		#non-iid
		elif self.data_dist == 'non-iid':
			#print('there')
			self.class_list = [class_[:min_cls_cnt] for class_ in self.class_list] # Trim samples to min. 
			print('class list',len(self.class_list[0]))
			
			# Classes > workers
			if self.num_classes > self.num_workers:				
				# Determine acceptable number of worker
				possible_num_workers = []
				for i in range(3,self.num_classes):
					if self.num_classes%i==0:
						possible_num_workers.append(i)
				print('pos_num', possible_num_workers)
				assert self.num_classes % self.num_workers == 0, f"For {self.dataset_name} classes > workers, pls choose num. workers from: {possible_num_workers}" # To ensure each worker have same num of classes
				for i in range(self.num_classes):
					temp_index = self.class_list[i] # Assigning whole class, but ensure each class have same num. samples
					self.worker_index[i%self.num_workers].extend(temp_index)
				print('worker index',len(self.worker_index[0]))
			
			# Classes = workers
			elif self.num_classes == self.num_workers:
				for i in range(self.num_classes):
					temp_index = self.class_list[i]
					self.worker_index[i%self.num_workers].extend(temp_index)
			
			# Classes < workers
			else:

				# Check if each worker can have same amount of data
				assert self.num_workers%self.num_classes == 0, f"For {self.dataset_name} workers > classes, pls choose num. workers from multiples of {self.num_classes}" # To ensure each worker have same num of classes

				start_id = [0 for _ in range(self.num_classes)] # start ID for each class
				class_worker_ratio = self.num_classes/self.num_workers # % of data per class for one worker
				assert class_worker_ratio<=0.5, "Warning: class_worker_ratio > 0.5 -- will cause heavy imbalance in number of data across different workers."

				# Compute sample per class to be assigned to single worker
				sample_per_class = int(np.floor(len(self.class_list[0])*class_worker_ratio)) # Classes in self.class_list should have all similar len

				for i in range(self.num_workers):
					class_ = i%self.num_classes # Determine assigned class for this worker
					temp_index = self.class_list[class_][start_id[class_]:start_id[class_]+sample_per_class] # Extract data from class
					self.worker_index[i].extend(temp_index) # Assign data to worker
					start_id[class_] += sample_per_class # Updates start ID for each class


		################################################################################### Ag data non-iid #######################################################################
		elif self.data_dist == 'non-iid-Ag':

			camera_layout = pd.read_excel('Camera_layout.xlsx')
			num_columns = int(self.num_cam/self.num_workers)
			column_choice = [*range(1,19-num_columns), *range(25, 43-num_columns)]
			start_column = random.choice(column_choice)
			start_row = random.randint(1,(14-self.num_workers))
			class_names = camera_layout.iloc[start_row:start_row+self.num_workers, start_column:start_column+num_columns].values.astype(int)
			# class_names = camera_layout.iloc[start_row, start_column:start_column+num_columns].values.astype(int)
			class_names = class_names.tolist()
			print(class_names)
			for j in range(self.num_workers):
				for each_agent in class_names[j]:
					num = '%03d' % each_agent
					string = 'CAM'+ str(num)
					i = self.data.classes.index(string)
					temp_index = self.class_list[i]
					self.worker_index[j].extend(temp_index)
				# print('i',i)
			# 	indices = [j for j, x in enumerate(self.data.targets) if x == i-1]
			# 	# print('indeces_tar', indices)
			# 	for p in indices:
			# 		self.new_labels_in.append(self.data.targets[p])
			# 		self.new_data_in.append(self.data[p])
			# print(self.new_labels_in)
			# print(self.new_data_in)
			# self.new_labels = np.asarray(self.new_labels_in)
			# classes, class_counts = np.unique(self.new_labels, return_counts=True)
			# self.num_classes = len(classes)
			# min_cls_cnt = min(class_counts)	
			# self.class_list = [[] for nb in range(self.num_classes)]
			# for class_ in classes:
			# 	self.class_list[class_] = np.argwhere(self.new_labels == class_).ravel()
			# self.class_list = [class_[:min_cls_cnt] for class_ in self.class_list] # Trim samples to min. 
			# # print('class list',len(self.class_list[0]))
			# # Classes > workers

			# possible_num_workers = []
			# for i in range(3,self.num_cam):
			# 	if self.num_classes%i==0:
			# 		possible_num_workers.append(i)
			# assert self.num_classes % self.num_workers == 0, f"For {self.dataset_name} classes > workers, pls choose num. workers from: {possible_num_workers}" # To ensure each worker have same num of classes
			# for i in classes:
			# 	temp_index = self.class_list[i] # Assigning whole class, but ensure each class have same num. samples
			# 	self.worker_index[i%self.num_workers].extend(temp_index)
			print('worker index',len(self.worker_index[0]))


################################################################################### Ag data non-iid-small #######################################################################
		elif self.data_dist == 'non-iid-Ag-small':
			self.class_list = [class_[:min_cls_cnt] for class_ in self.class_list] # Trim samples to min. 
			# print('class list',len(self.class_list))
			num_columns = int(self.num_cam/self.num_workers)
			class_names = [[139,862,414,43,728,339],[230,349,680,637,876,519],[78,287,166,227,864,752],[238,643,219,473,838,224],[302,82,787,687,235,733]]

			possible_num_workers = []
			for i in range(3,self.num_classes):
				if self.num_classes%i==0:
					possible_num_workers.append(i)
			# print('num_workers',possible_num_workers)
			assert self.num_classes % self.num_workers == 0, f"For {self.dataset_name} classes > workers, pls choose num. workers from: {possible_num_workers}" # To ensure each worker have same num of classes
			for d in range(self.num_workers):
				for each_agent in class_names[d]:
					num = '%03d' % each_agent
					string = 'CAM'+ str(num)
					i = self.data.classes.index(string)
					temp_index = self.class_list[i]
					self.worker_index[d].extend(temp_index)
					# print('worker index',len(self.worker_index[0]))
		
	
	def get_(self, rank_id):
		"""
		This function takes in a rank id and returns its share of minibatch
		"""
		return Partition(self.data, self.worker_index[rank_id])

						  
		 

def get_partition_dataloader(dataset, **kwargs):
	"""
	Partition the whole dataset into smaller sets for each rank.
	
	@param dataset: complete the dataset. We will 1) split it evenly to every worker and 2) build dataloaders on splitted dataset
	@batch_size: batch_size for the data loader! i.e. every worker will load this many samples 
	@param num_workers: the number of workers. Used to decide partition ratios
	@param myrank: the rank of this particular process
	@**kwargs partition: the partition ratio for dividing the data
	rvalue: training set for this particular rank
	"""
	data_dist = kwargs.get('data_dist', 'iid')
	batch_size = kwargs.get('batch_size', 128)
	num_workers = kwargs.get('num_workers')
	dataset_name = kwargs.get('data')
	myrank = kwargs.get('rank')
	if num_workers is None:
		raise ValueError('provide number of workers')

	partitioner = DataPartition(dataset, **kwargs)  # partitioner is in charge of producing shuffled id lists
	if kwargs.get('server_rank', -1) != -1:
		curr_rank_dataset = partitioner.get_(myrank-1)  # get the data partitioned for current rank, 0 is the server so -1
	else:
		curr_rank_dataset = partitioner.get_(myrank)  # get the data partitioned for current rank, 0 is the server so -1
	# build a dataloader based on the partitioned dataset for current rank
	train_set = DataLoader(curr_rank_dataset, batch_size=batch_size, shuffle=True)
	return train_set                          
 