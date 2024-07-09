# Exploring Deep Residual Learning for Image Recognition

This project implements various ResNet models (ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152) from scratch and trains them on CIFAR-10, CIFAR-100, and MNIST datasets. The performance of the models is evaluated using classification accuracy, precision, recall, and F1-score.

## File Structure
```
project/
├── datasets.py # Code for loading datasets
├── models.py # ResNet model definitions
├── train.py # Training and evaluation functions
└── main.py # Main script for training the models
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
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the necessary libraries:
    ```bash
    pip install torch torchvision scikit-learn
    ```

## Running the Code

To train a ResNet model on a specific dataset, use the `main.py` script with the appropriate arguments. Here are some examples:

1. Train ResNet-18 on CIFAR-10 for 10 epochs:
    ```bash
    python main.py --dataset CIFAR10 --model ResNet18 --epochs 10
    ```

2. Train ResNet-50 on CIFAR-100 for 10 epochs:
    ```bash
    python main.py --dataset CIFAR100 --model ResNet50 --epochs 10
    ```

3. Train ResNet-152 on MNIST for 10 epochs:
    ```bash
    python main.py --dataset MNIST --model ResNet152 --epochs 10
    ```
