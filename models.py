import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from Optimizers import *
from torch.optim.lr_scheduler import StepLR
import Models
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

resnet_list = {
    'resnet20': Models.resnet20,
    'resnet32': Models.resnet32,
    'resnet44': Models.resnet44,
    'resnet56': Models.resnet56,
    'resnet110': Models.resnet110,
    'resnet1202': Models.resnet1202
    }

vgg_list = ['VGG11', 'VGG13', 'VGG16', 'VGG19']

# class CNN(nn.Module):

#     def __init__(self,ch_dim=1, fc_nodes=576, num_classes=10):
#         super(CNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(ch_dim, 32, 3, padding=1),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(p=0.25),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#             )
#         self.classifier = nn.Linear(fc_nodes, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

class CNN(nn.Module):
    def __init__(self,ch_dim=3, fc_nodes=1024, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(ch_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(fc_nodes, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.ldrop = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.linear1(x)))
        x = self.ldrop(x)
        x = self.classifier(x)
        return x

class mnist_CNN(nn.Module):

    def __init__(self,ch_dim, num_classes=10):
        super(mnist_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            )
        self.classifier = nn.Linear(3136, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

### I need to fix this one
class VGG11_mnist(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(VGG11_mnist, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Big_CNN(nn.Module):

    def __init__(self,ch_dim, num_classes=10):
        super(Big_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FCN(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, nb_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        for _ in range(20):
            x = self.fc2(x)
        x = self.fc3(x)
        return x


class LR(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(LR, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, nb_classes, bias=False)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        return x


class aucifar(nn.Module):
    def __init__(self):
        super(aucifar, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # print('encode', encoded.shape)
        # print('decode', decoded.shape)
        # encode torch.Size([64, 48, 4, 4])
        # decode torch.Size([64, 3, 32, 32])
        return encoded, decoded	


class aumnist(nn.Module):
    def __init__(self):
        super(aumnist, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        # print('hereenc')
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())
        # print('heredec')
    def forward(self, x):
        # print('herefo')
        enc = self.encoder(x)
        # print('herefoenc')
        dec = self.decoder(enc)
        # print('herefodec')
        return enc, dec


class aumnist2(nn.Module):

    def __init__(self):
        super(aumnist2, self).__init__()

        cuda = True if torch.cuda.is_available() else False


        self.image_shape = (1,28,28)
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.encoder = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, 10)
        self.logvar = nn.Linear(512, 10)



        # print('hereenc')
        self.decoder = nn.Sequential(
            nn.Linear(10, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(self.image_shape))),
            nn.Tanh(),
        )
        # print('heredec')
    def forward(self, x):
        # print('herefo')
        img_flat = x.view(x.shape[0], -1)
        x1 = self.encoder(img_flat)
        mu = (self.mu(x1))
        logvar = self.logvar(x1)
        std = torch.exp(logvar / 2)
        # sampled_z = Variable(self.Tensor(np.random.normal(0, 1, (mu.size(0), 10))))
        sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), 10))))
        enc = sampled_z * std + mu
        # print('herefoenc')
        img_flat = self.decoder(enc)
        dec = img_flat.view(img_flat.shape[0], *self.image_shape)
        # print('herefodec')

        #print(enc,dec)
        return enc, dec

class aucifar_Res(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(aucifar_Res, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        # print('encode', z.shape)
        # print('decode', x_reconst.shape)
        # encode torch.Size([64, 256])
        # decode torch.Size([64, 3, 224, 224])
        return z, x_reconst

class supmnist(nn.Module):
    def __init__(self):
        super(supmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def BuildModel(model_name, img_dim, fc_nodes, nb_classes):

    loss = nn.CrossEntropyLoss()
    if model_name == "CNN":
        model = CNN(img_dim, fc_nodes, nb_classes)
    elif model_name == "supmnist":
        model = supmnist()
    elif model_name == "VGG11_mnist":
        model = VGG11_mnist()
    elif model_name == "Big_CNN":
        model = Big_CNN(img_dim, nb_classes)
    elif model_name == "FCN":
        model = FCN(img_dim, nb_classes)
    elif model_name == "LR":
        model = LR(img_dim, nb_classes)
    elif model_name == "PreResNet110" or model_name == "WideResNet28x10":
        model_cfg = getattr(Models, model_name)
        model = model_cfg.base(*model_cfg.args, num_classes=nb_classes, **model_cfg.kwargs)
    elif model_name in resnet_list.keys():
        model = resnet_list[model_name](nb_classes)
    elif model_name in vgg_list:
        model = Models.VGG(model_name, nb_classes)
    elif model_name == "mnist_CNN":
        model = mnist_CNN(img_dim, nb_classes)
    elif model_name == "aumnist":
        model = aumnist()
        loss = nn.MSELoss()
    elif model_name == "aucifar":
        model = aucifar()
        loss = nn.MSELoss()
    elif model_name == "aumnist2":
        model = aumnist2()
        loss = nn.MSELoss()
    elif model_name == "aucifar_Res":
        model = aucifar_Res()
        loss = nn.MSELoss()

    return model,loss



def LoadModel(img_dim, **kwargs):
    #  Get required meta data
    wsize = kwargs.get('wsize', -1)
    assert wsize > -1
    myrank = kwargs.get('rank', -1)
    assert myrank > -1
    assert myrank < wsize
    #  Criterion for dealing with federated learning and collaborative learning both
    #  for collaborative learning server_rank will be -1
    server_rank = kwargs.get('server_rank',-1)
    num_workers = kwargs.get('num_workers', wsize)
    device = kwargs.get('device')

    modelkey = kwargs['model_arch']
    optkey = kwargs['optimizer']

    # Setting params for CNN:
    if kwargs["data"] == "MNIST" or kwargs["data"] == 'semeion':
        ch_dim = 1
        fc_nodes = 576
        num_classes = 10
    elif kwargs["data"] == "CIFAR10":
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 10
    elif kwargs["data"] == "CIFAR100":
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 100
    elif kwargs["data"] == "ImageNet":
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 1000
    elif kwargs["data"] == 'STL10':
        fc_nodes = 9216
    elif kwargs["data"] in ["Agdata","Agdata-small"]:
        ch_dim = 3
        fc_nodes = 1024
        num_classes = 1

    if modelkey == 'LR':
        ch_dim = img_dim[0]*img_dim[1]*img_dim[2]

    # Build model
    model, criterion = BuildModel(modelkey, ch_dim, fc_nodes, num_classes)
    model = model.to(device)


    # initialize criterion
    # criterion = nn.CrossEntropyLoss()

    # initialize parameters
    for param in model.parameters():
        param.grad = torch.zeros(param.size(), requires_grad=True).to(device)
        param.grad.data.zero_()

    # initialize optimizer
    if optkey=='nesterov':
        optimizer = optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9), nesterov=True)
    elif optkey=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.0001), momentum=kwargs.get('momentum', 0.9), nesterov=False)
    elif optkey=='adam':
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey=='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=kwargs.get('lr', 1.0))
    elif optkey=='CDSGD':
        optimizer = CDSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey=='CDMSGD':
        optimizer = CDMSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey=='DSMA':
        optimizer = DSMA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SGA':
        optimizer = SGA(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optkey == 'CompLGA':
        optimizer = CompLGA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'LGA':
        optimizer = LGA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'CompLGA':
        optimizer = CompLGA(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SGP':
        optimizer = SGP(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'SwarmSGD':
        optimizer = SwarmSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))
    elif optkey == 'LDSGD':
        optimizer = LDSGD(model.parameters(), kwargs, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9))

    # Implement LR scheduler
    if kwargs['scheduler']:
        scheduler = StepLR(optimizer, step_size=kwargs['LR_sche_step'], gamma=kwargs['LR_sche_lamb'])
        if kwargs['verbose'] >= 1:
            if kwargs['dist'].get_rank() == 0:
                print("Applying Step LR scheduler with step_size = %s and lr_lambda = %s"%(kwargs['LR_sche_step'], kwargs['LR_sche_lamb']))
    else:
        scheduler = None
        if kwargs['dist'].get_rank() == 0:
            print('Not applying LR scheduler.')

    return model, optimizer, criterion, scheduler
