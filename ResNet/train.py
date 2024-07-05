import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

import tqdm
from tqdm.auto import trange

from ResNet.resnet import *


def resnet_train_cifar10(batch_size=50, learning_rate=0.0002, num_epoch=100, device="cpu", blocks=50):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # define dataset
    cifar10_train = datasets.CIFAR10(root="./Data/", train=True, transform=transform, target_transform=None, download=True)
    cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

    # define loader
    train_loader = DataLoader(cifar10_train,batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(cifar10_test,batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device(device)
    model = None
    if blocks == 18:
        model = ResNet18().to(device)
    if blocks == 34:
        model = ResNet34().to(device)
    if blocks == 50:
        model = ResNet50().to(device)
    if blocks == 101:
        model = ResNet101().to(device)
    if blocks == 152:
        model = ResNet152().to(device)

    summary(model, (3, 32, 32))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_arr = []
    for i in trange(num_epoch):
        for j,[image,label] in enumerate(train_loader):
            x = image.to(device)
            y_= label.to(device)
            
            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

        if i % 10 ==0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())