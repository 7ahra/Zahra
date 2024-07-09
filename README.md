# Exploring Deep Residual Learning for Image Recognition

This project implements various ResNet models (ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152) from scratch and trains them on CIFAR-10, CIFAR-100, and MNIST datasets. The performance of the models is evaluated using classification accuracy, precision, recall, and F1-score.

## File Structure
```
project/
└──resnet
    ├── block_config.py
    ├── __init__.py
    ├── models.py
    └── train.py
└──utils
    ├── datasets.py
    ├── evaluations.py
    └── log_texts.py
└── main.py
├── README.md
└── requirements.txt

```


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohtasimhadi/exploring_deep_residual_learning_for_image_recognition.git
    cd exploring_deep_residual_learning_for_image_recognition
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

```bash
python main.py --dataset <dataset_name> --layers <num_layers> [--epochs <num_epochs>] [--batch_size <batch_size>]
```

### Arguments:
- dataset: Specify the dataset to use for training.
  Choices: `'CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet'`
- layers: Number of layers in the ResNet model.
- epochs: Number of epochs to train the model (default: 10).
- batch_size: Batch size for training and validation (default: 16).

### Example Commands:
Train ResNet on CIFAR-10 with 50 layers for 20 epochs:
```bash
python main.py --dataset CIFAR10 --layers 50 --epochs 20 --batch_size 32
```
Train ResNet on ImageNet with 101 layers for 50 epochs:
```bash
python main.py --dataset ImageNet --layers 101 --epochs 50 --batch_size 64
```
### Notes:
Adjust --batch_size according to your system's GPU memory capacity.
The script will download the datasets automatically (except ImageNet) if they are not found in the specified directory. Download ImageNet dataset from https://www.image-net.org/download.php
