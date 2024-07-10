import argparse
from torch.utils.data import DataLoader
from utils import get_datasets
from resnet import get_model, run_training

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10, MSCOCO, or ImageNet')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'MSCOCO', 'ImageNet'], help='Dataset to use')
    parser.add_argument('--layers', type=int, required=True, help='Number of ResNet layers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_dataset, test_dataset, num_classes = get_datasets(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    run_training(model_fn=get_model, train_loader=train_loader, test_loader=test_loader, num_classes=num_classes, num_layers=args.layers, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
