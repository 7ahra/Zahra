import argparse
from torch.utils.data import DataLoader
from utils.datasets import get_datasets
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from resnet.train import run_training

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10, CIFAR-100, or MNIST')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'MNIST'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'], help='Model to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    train_dataset, test_dataset, num_classes = get_datasets(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_dict = {
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152
    }

    model_fn = model_dict[args.model]
    run_training(model_fn, train_loader, test_loader, num_classes, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
