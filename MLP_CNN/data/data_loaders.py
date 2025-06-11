from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

#Simple MLP/CNN, no need to resize to 224, 224

transform_mnist = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_cifar = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_loaders(dataset, batch_size=64):
    if dataset == 'MNIST':
        data_path = './data/data_MNIST'
        print(f"MNIST directory contents: {os.listdir(data_path) if os.path.exists(data_path) else 'Directory does not exist'}")
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                print(f"{root}: {files}")
        
        train_set = datasets.MNIST('./data/data_MNIST', train=True, download=True, transform=transform_mnist)
        test_set = datasets.MNIST('./data/data_MNIST', train=False, transform=transform_mnist)
        in_channels = 1
        
    elif dataset == 'CIFAR10':
        data_path = './data/data_CIFAR'
        print(f"CIFAR directory contents: {os.listdir(data_path) if os.path.exists(data_path) else 'Directory does not exist'}")
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                print(f"{root}: {files}")
        
        train_set = datasets.CIFAR10('./data/data_CIFAR', train=True, download=True, transform=transform_cifar)
        test_set = datasets.CIFAR10('./data/data_CIFAR', train=False, transform=transform_cifar)
        in_channels = 3
    else:
        print(dataset)
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_channels

