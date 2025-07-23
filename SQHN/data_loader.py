import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pickle

# ======================================================================================================================
# This file contains the data loaders for the continual learning tasks.
# ======================================================================================================================

# Define transforms
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Online data loader
def get_data(shuf, data, max_iter, cont=False, b_sz=1):
    # Get datasets
    if data == 0: # MNIST
        train_data = torchvision.datasets.MNIST('data/', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.MNIST('data/', train=False, download=True, transform=trans)
    elif data == 1: # F-MNIST
        train_data = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=trans)
    elif data == 2: # CIFAR-10
        train_data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=trans)
    elif data == 3: # CIFAR-100
        train_data = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.CIFAR100('data/', train=False, download=True, transform=trans)
    elif data == 4: # SVHN
        train_data = torchvision.datasets.SVHN('data/', split='train', download=True, transform=trans)
        test_data = torchvision.datasets.SVHN('data/', split='test', download=True, transform=trans)
    elif data == 5: # Tiny ImageNet
        # Asumiendo que ya has descargado y descomprimido Tiny ImageNet en 'data/tiny-imagenet-200'
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', transform=trans)
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', transform=trans)
    elif data == 6: # EMNIST
        train_data = torchvision.datasets.EMNIST('data/', split='balanced', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.EMNIST('data/', split='balanced', train=False, download=True, transform=trans)

    if cont:
        # Create a dictionary to store indices for each class
        class_indices = {i: [] for i in range(len(train_data.classes))}
        for idx, (_, label) in enumerate(train_data):
            class_indices[label].append(idx)
        
        # Create a balanced list of indices for continual learning
        continual_indices = []
        num_samples_per_class = max_iter // len(train_data.classes)
        for i in range(len(train_data.classes)):
            continual_indices.extend(class_indices[i][:num_samples_per_class])
        
        # Adjust if total is less than max_iter due to rounding
        if len(continual_indices) < max_iter:
            remaining = max_iter - len(continual_indices)
            # Add remaining samples from the first class
            continual_indices.extend(class_indices[0][num_samples_per_class:num_samples_per_class + remaining])
            
        train_subset = torch.utils.data.Subset(train_data, continual_indices)
        train_loader = DataLoader(train_subset, shuffle=shuf, batch_size=b_sz)
    else:
        # Standard loader if not continual
        train_loader = DataLoader(train_data, shuffle=shuf, batch_size=b_sz)

    test_loader = DataLoader(test_data, shuffle=False, batch_size=1000)
    return train_loader, test_loader

# Data loader for domain incremental learning (MNIST-FMNIST, CIFAR-SVHN)
def get_dom_train_loader(shuf, online, iter_dom):
    # MNIST and F-MNIST
    mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True, transform=trans)
    fmnist_train = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=trans)
    
    # CIFAR and SVHN
    cifar_train = torchvision.datasets.CIFAR100('data/', train=True, download=True, transform=trans)
    svhn_train = torchvision.datasets.SVHN('data/', split='train', download=True, transform=trans)

    # Combine datasets
    if online:
        # Interleave samples
        combined_dataset = torch.utils.data.ChainDataset([mnist_train, fmnist_train, cifar_train, svhn_train])
    else:
        # Present datasets sequentially
        mnist_subset = torch.utils.data.Subset(mnist_train, range(iter_dom))
        fmnist_subset = torch.utils.data.Subset(fmnist_train, range(iter_dom))
        cifar_subset = torch.utils.data.Subset(cifar_train, range(iter_dom))
        svhn_subset = torch.utils.data.Subset(svhn_train, range(iter_dom))
        combined_dataset = torch.utils.data.ConcatDataset([mnist_subset, fmnist_subset, cifar_subset, svhn_subset])

    train_loader = DataLoader(combined_dataset, shuffle=shuf, batch_size=1)
    return train_loader