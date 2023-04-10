import os
import numpy as np
import utils
import models
import data
from federated import Federated
import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
torch.cuda.set_device(0)
torch.manual_seed(0)

if __name__ == '__main__':
    
    # initialize MPI environment 
    dist.init_process_group(backend="mpi")
    myrank = dist.get_rank()
    wsize = dist.get_world_size()  # number of processes = num_workers + 1   
    server_rank = 0 # this process is the server
    
    #############################################################################################################################
    #                                          setup code shared by workers and server
    #############################################################################################################################

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--use_cuda', type=str, default='true', help='Use CUDA if available')
    parser.add_argument('--data',type=str,default='MNIST', help='Define the data used for training')
    parser.add_argument('--model',type=str,default='CNN', help='Define the model used for training')
    parser.add_argument('--opt', type=str, default='SGD',
                        help='Optimizer (Examples: adam, rmsprop, adadelta, ...)')
    parser.add_argument('--batch_size', type=int, default=128, help='Define batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Define num epochs for training')

    args = parser.parse_args()
    
    torch.manual_seed(123)
    num_workers = wsize-1
    batch_size=args.batch_size
    num_epochs = args.epochs
    device = torch.device("cuda" if args.use_cuda == 'true' else "cpu")

    argdict = {
    'use_cuda':args.use_cuda,
    'data':args.data,
    'model_arch':args.model,
    'data_dist':'iid',
    'server_rank':server_rank,
    'epochs':args.epochs,
    'batch_size':args.batch_size,
    'optimizer':args.opt,
    'device':device,
    'wsize':wsize,
    'num_workers':num_workers,
    'rank':myrank}

    #############################################################################################################################
    #                                                 workers' setup code
    ############################################################################################################################# 
    dataloader = data.LoadData(args.data, dist, **argdict)
    model, opt, criterion = models.LoadModel(args.model, args.opt, dist, **argdict) # workers and servers all have a model! 
    dist.barrier()
    trainer = Federated(dataloader, model, opt, criterion, dist, **argdict)
    trainer.train()
