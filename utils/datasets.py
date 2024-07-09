import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

def get_datasets(dataset_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
        num_classes  = 10
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
        test_dataset  = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform_mnist)
        test_dataset  = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform_mnist)
        num_classes = 10
    elif dataset_name == "ImageNet":
        train_dataset = torch.utils.data.DataLoader(ImageFolder('./datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train', transform=transform), batch_size=16, shuffle=True)
        test_dataset  = torch.utils.data.DataLoader(ImageFolder('./datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test', transform=transform), batch_size=16, shuffle=False)
        num_classes = 1000
    else:
        raise ValueError("Unsupported dataset. Please choose from 'CIFAR10', 'CIFAR100', 'MNIST', or 'ImageNet'.")

    return train_dataset, test_dataset, num_classes
