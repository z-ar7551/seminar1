Barlow Twins: Self-Supervised Learning via Redundancy Reduction
---------------------------------------------------------------

PyTorch implementation of [Barlow Twins](https://arxiv.org/abs/2103.03230) on the SSL4EO-S12 v1.1 dataset

```
@article{zbontar2021barlow,
  title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
  author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, St{\'e}phane},
  journal={arXiv preprint arXiv:2103.03230},
  year={2021}
}
````
### Setup
Download the dataset and unzip the files.
Run the make_dataset.py file to convert the .zarr files to .png
```
python make_dataset.py
```
Arrange the resulting images in the following format:
barlow_data/train/class_name/img_num.png

### Barlow Twins Training
Pre-train the model
```
python main.py barlow_data
```
adjust hyperparameters according to the amount of data and expected results.

### Evaluation: Linear Classification

Train a linear probe on the representations learned by Barlow Twins. Freeze the weights of the resnet on any training set (Eg. EuroSAT).

```
python evaluate.py EuroSAT/train checkpoint/resnet50.pth --lr-classifier 0.3
```

### Evaluation: Semi-supervised Learning

Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet and use any training set training set.

```
python evaluate.py EuroSAT/train checkpoint/resnet50.pth --weights finetune --epochs 20 --lr-backbone 0.005 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir ./checkpoint/semisup/
```

