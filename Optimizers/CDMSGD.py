import torch
from torch.optim import Optimizer
from collections import defaultdict
import numpy as np
import time

class CDMSGD(Optimizer):
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
		super(CDMSGD, self).__init__(params, defaults)

		# Extract required var.
		self.pi_all = kwargs.get('pi')
		self.rank = kwargs.get('rank')
		self.kwargs = kwargs
		self.pi = kwargs.get('pi') # pi based on current rank
		self.device = kwargs.get('device')

		self.local_neigh = np.argwhere(np.asarray(self.pi) != 0.0).ravel() # Find connected agents # give indices of local neighbors for each rank

		#print("local_neigh =", self.local_neigh)
		self.pi = [self.pi[i] for i in self.local_neigh] # simplified pi -- with only connected agents # just the pi elements that have value in each rank
		#print("self.pi =", self.pi)


	def __setstate__(self, state):
		super(CDMSGD, self).__setstate__(state)
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

			for j, p in enumerate(group['params']):

				if p.grad is None:
					continue

				d_p = p.grad.data # g_t
				#print("data size =", p.data.size())
				con_buf = [torch.zeros(p.data.size()).to(self.device) for _ in range(len(neighbors))] # Parameters placeholder
				
				dist.all_gather(con_buf, p.data, group=distgroup) # Gather parameters from workers to con_buf [param worker 0, param worker 1, ...]

				buf = torch.zeros(p.data.size()).to(self.device)
				
				# Extract connected agents data only
				con_buf = [con_buf[i] for i in self.local_neigh] #based on the current rank

				for pival, con_buf_agent in zip(self.pi, con_buf):
					#kind of averaging the parameters of connected neighbors in the current rank (for current agent)
					# to get teta_t-1
					buf.add_(other=con_buf_agent, alpha=pival) # multiply other*alpha(multiplier for other- elements of w matrix) and add it to buf

				param_state = self.state[p]
				if 'momentum_buffer' not in param_state:
					m_buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).to(self.device)
					m_buf.mul_(momentum).add_(d_p) # b_t := momentum* b_t-1 + g_t
				else:
					m_buf = param_state['momentum_buffer']
					m_buf.mul_(momentum).add_(other=d_p, alpha=1-dampening) # b_t := momentum* b_t-1 + (1-damping)* g_t

				d_p.add_(other=m_buf, alpha=momentum) #g_t := g_t-1 + momentum* b_t
				p.data = buf.add_(other=d_p, alpha=-group['lr']) #teta_t := teta_t-1 - lr*g_t
				
		return loss
