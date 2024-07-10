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

## Downloading Datasets
**ImageNet**
If ImageNet dataset is not downloaded, use the following code to download and extract it.
```bash
python downloader.py imageNet
```

**MS COCO**
If MS COCO dataset is not downloaded, use the following code to download and extract it.
```bash
python downloader.py ms_coco
```

### Notes
- CIFAR10 dataset will be downloaded automatically during the training.
- If download fails due to url issues, the urls can be changed from `utils/urls.json`

## Running the Code

```bash
python main.py --dataset <dataset_name> --layers <num_layers> [--epochs <num_epochs>] [--batch_size <batch_size>]
```

### Arguments:
- **dataset:** Specify the dataset to use for training.

  *choices:* `CIFAR10`, `MSCOCO`, `ImageNet`
- **layers:** Number of layers in the ResNet model.
- **epochs:** Number of epochs to train the model (default: 10).
- **batch_size:** Batch size for training and validation (default: 16).

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
- Adjust `--batch_size` according to your system's GPU memory capacity.
- The script will download the datasets automatically (except ImageNet) if they are not found in the specified directory. Download ImageNet dataset from [ImageNet Website](https://www.image-net.org/download.php).


## References
```tex
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

@article{deng2009imagenet,
  title={ImageNet: A Large-Scale Hierarchical Image Database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  journal={IEEE Computer Vision and Pattern Recognition},
  year={2009}
}

@techreport{cifar10,
  author = {Krizhevsky, Alex and Hinton, Geoffrey},
  title = {Learning Multiple Layers of Features from Tiny Images},
  institution = {University of Toronto},
  year = {2009},
  type = {Technical Report}
}

@article{lecun1998mnist,
  author = {LeCun, Yann and Cortes, Corinna and Burges, CJ.C.},
  title = {{MNIST} handwritten digit database},
  journal = {AT\&T Labs},
  year = {1998},
  note = {\url{http://yann.lecun.com/exdb/mnist/}}
}

@article{everingham2010pascal,
  author = {Everingham, Mark and Van Gool, Luc and Williams, Christopher K. I. and Winn, John and Zisserman, Andrew},
  title = {The {PASCAL} Visual Object Classes Challenge},
  journal = {International Journal of Computer Vision},
  volume = {88},
  number = {2},
  pages = {303--338},
  year = {2010}
}

@inproceedings{lin2014microsoft,
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  title = {{Microsoft COCO}: Common Objects in Context},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2014}
}

```
