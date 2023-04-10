import torch
from trainer import Trainer

class Federated(Trainer):
    """docstring for Federated"""
    def __init__(self, dataloader, model, opt, criterion, dist, **kwargs):
        super(Federated, self).__init__(dataloader, model, opt, criterion, dist, **kwargs)
        self.server_rank = kwargs.get('server_rank', -1)
        self.num_workers = kwargs.get('num_workers', -1)
        assert self.wsize == self.num_workers + 1, 'num_workers is one less than total no. of agents, server_rank is not -1'
        assert self.num_workers > 0, 'num_workers must be greater than zero'
        self.workers = [i for i in range(self.wsize) if i is not self.server_rank]
        self.distgroup = dist.new_group(ranks=self.workers)
        self.servergroup = dist.new_group(ranks=[self.server_rank])
        self.hand_shake()

    def hand_shake(self):
        if not self.server_rank is -1:
            if self.rank is not self.server_rank:
                numdata = torch.tensor(len(self.dataloader))
                # all reduce phase 1
                self.dist.send(numdata, dst=self.server_rank, tag = 10+self.rank)
            elif self.rank is self.server_rank:
                numdata = [torch.tensor(0) for _ in range(self.num_workers)]
                for idx, w in enumerate(self.workers):
                    self.dist.recv(tensor=numdata[idx], src=w, tag=10+w)
            self.numdata = numdata
        else:
            raise ValueError('server_rank must not be -1 for Federated')


    def federate(self):
        for name, param in self.model.named_parameters():
            param_tensor = param.data
            if self.rank == self.server_rank:
                param_tensor = param.data
            self.dist.reduce(tensor=param_tensor, dst=self.server_rank, op=self.dist.ReduceOp.SUM)
            if (self.rank == self.server_rank):
                # receive params 
                param_tensor = torch.div(param_tensor.to(self.device), self.num_workers)
            # send the information back to the worker who send in gradients
            # the server send parameters to all workers 
            self.dist.broadcast(param_tensor, self.server_rank)
            param.data = param_tensor.to(self.device)



    def train_epoch(self, epoch_id):
        if not (self.rank == self.server_rank):
            # worker code
            epoch_loss = 0.0
            self.num_samples_seen = 0
            for batch_idx, (data, target) in enumerate(self.dataloader):  
                # initialize optimizer
                self.optimizer.zero_grad()
                # compute the gradient first or else gradient will be None.
                output = self.model(data.to(self.device))
                loss = self.criterion(output, target.to(self.device))
                loss.backward()
                epoch_loss += loss.item()
                self.num_samples_seen += len(target)
                # update 
                self.optimizer.step()
            self.worker_loss_hist.append(epoch_loss/self.num_samples_seen)
        self.federate()
