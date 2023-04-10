### This is the DSMA + Federated version for non dominating neighbors
# pytorch tutorial:https://pytorch.org/tutorials/intermediate/dist_tuto.html 
import torch
from torch.optim import Optimizer
from collections import defaultdict
import numpy as np
import time

class DSMA(Optimizer):
	def __init__(self, params, kwargs, lr=0.01, momentum=0.95, dampening=0,
				 weight_decay=0):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if not 0.0 <= weight_decay:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay)
		if (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(DSMA, self).__init__(params, defaults)

		# Extract required var.
		self.rank = kwargs.get('rank')
		self.kwargs = kwargs
		self.pi_all = kwargs.get('pi') # pi based on current rank
		self.pi_dom = kwargs.get('pi_dom')
		self.device = kwargs.get('device')
		self.DS = kwargs.get('DS')

		#print(self.pi_dom)
		self.local_neigh = np.argwhere(np.asarray(self.pi_all) != 0.0).ravel() # Find connected agents # give indices of local neighbors for each rank
		#print(self.rank, self.local_neigh)
		self.non_dom_neighbors = [neighbor for neighbor in self.local_neigh if neighbor not in self.DS]
		self.dom_neighbors = [neighbor for neighbor in self.local_neigh if (neighbor in self.DS and neighbor!= self.rank)]
		#print(self.rank, self.non_dom_neighbors, self.dom_neighbors)
		self.pi = [self.pi_all[i] for i in self.local_neigh] # simplified pi -- with only connected agents # just the pi elements that have value in each rank
		if self.rank in self.DS:
			self.pi_rank = self.pi_dom[list(self.DS).index(self.rank)][list(self.DS).index(self.rank)]
			self.pi_dom = [self.pi_dom[list(self.DS).index(self.rank)][list(self.DS).index(i)] for i in self.dom_neighbors] # simplified pi_dom -- with only connected agents # just the pi_dom elements that have value in each rank
			
	def __setstate__(self, state):
		super(DSMA, self).__setstate__(state)
		for group in self.param_groups:

			#print("one group of parameters =", self.group)
			group.setdefault('nesterov', True)

	# def step(self, closure=None):
	# def step(self, opt_kwargs, closure=None):
	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""

		# # Extract var. added thru Collab()
		dist = self.kwargs['dist']
		distgroup = self.kwargs['distgroup']
		neighbors = self.kwargs['neighbors']

		loss = None
		if closure is not None:
			loss = closure()
		if not isinstance(self.state, defaultdict):
			self.state = defaultdict(dict)

		for i, group in enumerate(self.param_groups):## Update rule
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']

			sum_timecomp_para = 0
			sum_timecomm_para = 0
			for j, p in enumerate(group['params']):

				#print("index of parameter =", j)
				#print("size of parameter =", p.size())
				if p.grad is None:
					continue

				time1sa = time.process_time()
				d_p = p.grad.data # g_t

				con_buf = [torch.zeros(p.data.size()).to(self.device) for _ in range(len(neighbors))] # Parameters placeholder

				timescomm = time.process_time()
				dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf [param worker 0, param worker 1, ...]
				timecomm = time.process_time() - timescomm

				if self.rank in self.DS: #example (5 agents ring): [0,1,2]

					#non dominating nodes send their model parameters to dominating nodes
					#print(self.rank, self.dom_neighbors, self.non_dom_neighbors) # example: 0 [1] [4]
					if self.non_dom_neighbors:
						for neighbor in self.non_dom_neighbors:
							p.data.add_(con_buf[neighbor]) #p.data will be also updated in con_buf
						p.data.mul_(1/(len(self.non_dom_neighbors)+1)) #averaging gradients of dom node neighbors #p.data will be also updated in con_buf


				#federated part
				if self.rank not in self.DS:
					for neighbor in self.dom_neighbors:
						p.data.add_(con_buf[neighbor])
					p.data.mul_(1/(len(self.dom_neighbors)+1))


				if self.rank in self.DS: 
					con_buf[self.rank] = p.data	
					p.data.mul_(self.pi_rank)
					for pival, dom_neigh in zip(self.pi_dom, self.dom_neighbors): #kind of averaging the parameters of connected neighbors in the current rank (for current agent) based on pi matrix # to get teta_t-1
						p.data.add_(other=con_buf[dom_neigh], alpha=pival) # multiply other*alpha(multiplier for other- elements of w matrix) and add it to buf
					#con_buf[self.rank] = p.data # Dont need to have this line since .add_


				param_state = self.state[p]
				if 'momentum_buffer' not in param_state:
					m_buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).to(self.device)
					m_buf.mul_(momentum).add_(d_p) # b_t := momentum* b_t-1 + g_t
				else:
					m_buf = param_state['momentum_buffer']
					m_buf.mul_(momentum).add_(other=d_p, alpha=1-dampening) # b_t := momentum* b_t-1 + (1-damping)* g_t
				d_p.add_(other=m_buf, alpha=momentum) #g_t := g_t-1 + momentum* b_t
				p.data = p.data.add_(other=d_p, alpha=-group['lr']) #teta_t := teta_t-1 - lr*g_t

		return loss


# SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2 singularity exec --nv -B /data:/data /data/py_1_7_tfbase.img python3.7 -m torch.distributed.launch --nnodes 1 --nproc_per_node 5 main.py --use_cuda --dom_CDMSGD --opt DSMA --experiment 1 --epochs 200