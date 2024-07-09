import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.log_texts import LOG, ERROR

def get_datasets(dataset_name):
    print(f"{LOG}Getting the dataset ready!")
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
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform_mnist)
        test_dataset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform_mnist)
        num_classes = 10
    else:
        raise ValueError(f"{ERROR} Unsupported dataset. Please choose from 'CIFAR10', 'CIFAR100', or 'MNIST'.")

    return train_dataset, test_dataset, num_classes
