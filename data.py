from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

#transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()]) ### F: This "transformations" is not used in the code.

def DataGenTrain(key):
    if key == 'MNIST':
        yield datasets.MNIST('../../shared/data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    elif key == 'CIFAR10':
        yield datasets.CIFAR10('../../shared/data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # From PyTorch Tutorial: 
                                   # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
                                   # transforms.Normalize((0.5, 0.5, 0.5),
                                   #                      (0.5, 0.5, 0.5)),
                                   # From cdshd Repo:
                                   # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        # (0.2023, 0.1994, 0.2010)),
                                   # From Pytorch Forum:
                                   # https://github.com/kuangliu/pytorch-cifar/issues/19
                                   # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.247, 0.243, 0.261)),
                               ]))
    elif key == 'CIFAR100':
        yield datasets.CIFAR100('../../shared/data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))

    elif key == 'ImageNet':
        yield datasets.ImageFolder('./data/imagenet/train',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   #transforms.RandomResizedCrop(224),
                                   transforms.RandomResizedCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225)),
                               ]))

    elif key == 'Agdata-small':
        yield datasets.ImageFolder('../../shared/data/Non_IID',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),transforms.Resize((128,128),interpolation=3),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.247, 0.243, 0.261)),
                                ]))
    elif key == 'Agdata':
        yield datasets.ImageFolder('../../shared/data/all-data/downsized.1030x773',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),transforms.Resize((128,128),interpolation=3),
                                #    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                #                         (0.247, 0.243, 0.261)),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                                ]))




def DataGenTest(key):
    if key == 'MNIST':
        yield datasets.MNIST('../../shared/data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    elif key == 'CIFAR10':
        yield datasets.CIFAR10('../../shared/data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # From PyTorch Tutorial: 
                                   # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
                                   # transforms.Normalize((0.5, 0.5, 0.5),
                                   #                      (0.5, 0.5, 0.5)),
                                   # From cdshd Repo:
                                   # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        # (0.2023, 0.1994, 0.2010)),
                                   # From Pytorch Forum:
                                   # https://github.com/kuangliu/pytorch-cifar/issues/19
                                   # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.247, 0.243, 0.261)),
                               ]))
    elif key == 'CIFAR100':
        yield datasets.CIFAR100('../../shared/data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))

    elif key == 'ImageNet':
        yield datasets.ImageFolder('./data/imagenet/valid',
                               transform=transforms.Compose([
                                   # transforms.Resize(256),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Resize(36),
                                   transforms.CenterCrop(32),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225)),
                               ]))

    elif key in ['Agdata', 'Agdata-small']:
        yield datasets.ImageFolder('../../shared/data/AE_images_30cameras/Validation',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),transforms.Resize((128,128),interpolation=3),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                                ]))
    
    


DataYielderTrain = lambda key : next(DataGenTrain(key))
DataYielderTest = lambda key : next(DataGenTest(key))

def LoadData(**kwargs):
    # train_loader = None
    wsize = kwargs.get('wsize', -1) # num agents e.g. 5
    assert wsize > -1
    myrank = kwargs.get('rank', -1) # current rank 
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    if not server_rank in list(range(wsize)): #server_rank = -1
        assert num_workers == wsize
    batch_size = kwargs.get('batch_size', 128)
    if (myrank != server_rank):
        dataset = DataYielderTrain(kwargs['data'])
        # if kwargs['data'] == 'Agdata':
        #     num_train = len(dataset)
        #     # print(num_train)
        #     indices = list(range(num_train))
        #     # print(indices)
        #     np.random.shuffle(indices)
        #     # print(indices)
        #     train_sampler = SubsetRandomSampler(indices)
        #     # print(train_sampler)
        #     train_loader = DataLoader(dataset, kwargs['batch_size'], sampler=train_sampler)
        # else:
        whole_data = DataLoader(dataset, kwargs['batch_size'], shuffle=True)
        train_loader = utils.get_partition_dataloader(dataset, **kwargs)
        print(kwargs['data'], 'Loaded and Partitioned, total training samples', len(whole_data.dataset))
        print(kwargs['data'], 'Loaded and Partitioned, training samples for worker rank {}: '.format(myrank), len(train_loader.dataset))
    return train_loader


def LoadTestData(**kwargs):
    # test_loader = None
    wsize = kwargs.get('wsize', -1)
    assert wsize > -1
    myrank = kwargs.get('rank', -1)
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    if not server_rank in list(range(wsize)):
        assert num_workers == wsize
    batch_size = kwargs.get('batch_size', 128)
    if (myrank != server_rank):
        test_data = DataYielderTest(kwargs['data'])
        # if kwargs['data'] == 'Agdata':
        #     num_test = len(test_data)
        #     # print(num_train)
        #     indices = list(range(num_test))
        #     # print(indices)
        #     np.random.shuffle(indices)
        #     # print(indices)
        #     test_sampler = SubsetRandomSampler(indices)
        #     # print(train_sampler)
        #     test_loader = DataLoader(test_data, kwargs['batch_size'], sampler= test_sampler)
        # else:
        test_loader = DataLoader(test_data, kwargs['batch_size'], shuffle=True)
        print(kwargs['data'], 'Loaded and Partitioned, total testing samples for worker rank {}: '.format(myrank), len(test_loader.dataset))
        data_dim = np.shape(test_data[0][0].numpy())
    return test_loader, data_dim

