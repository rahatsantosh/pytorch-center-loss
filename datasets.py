import torch
import torchvision
from torch.utils.data import DataLoader
import os

import transforms

class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False
        path = '/content/drive/My Drive/fellowship'
        # data path
        datapath = path
        # veggie data path
        data_folder = 'Data_07_preprocessing/training'
        veggiedatapath = os.path.join(datapath,data_folder)

        trainset = torchvision.datasets.ImageFolder(
            root=veggiedatapath,
            transform=transform
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = torchvision.datasets.ImageFolder(
            root='/content/drive/My Drive/fellowship/new_test',
            transform=transform
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

__factory = {
    'mnist': MNIST,
}

def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)
