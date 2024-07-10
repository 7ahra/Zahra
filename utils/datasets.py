import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_datasets(dataset_name):

    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
        num_classes  = 10
    elif dataset_name == 'MSCOCO':
        train_dataset = datasets.CocoDetection(root='./datasets/ms_coco/train2017', annFile='./datasets/ms_coco/annotations/instances_train2017.json', transform=None)
        test_dataset  = datasets.CocoDetection(root='./datasets/ms_coco/test2017',  annFile='./datasets/ms_coco/annotations/image_info_test2017.json',transform=None)
        num_classes = 80
    elif dataset_name == "ImageNet":
        train_dataset = ImageFolder('./datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train', transform=transform)
        test_dataset  = ImageFolder('./datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test', transform=transform)
        num_classes = 1000
    else:
        raise ValueError("Unsupported dataset. Please choose from 'CIFAR10', 'CIFAR100', 'MNIST', or 'ImageNet'.")

    return train_dataset, test_dataset, num_classes
