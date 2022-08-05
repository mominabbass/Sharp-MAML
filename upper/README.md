# Sharp-MAML_up

An implementation of Sharp-MAML in [PyTorch](https://pytorch.org/) with [Torchmeta](https://github.com/tristandeleu/pytorch-meta).

#### Requirements
* Python: 3.9.7
* Pytorch: 1.11.0

### Preparation
Download the data from [this link](https://drive.google.com/drive/folders/1OT8mNSKoTvhgT3dE1g545LrPuUjLVcfJ?usp=sharing). Unzip the datasets and place them into a folder.

### Usage
You can use [`train_mini-sharp.py`](train_mini-sharp.py) to meta-train your model with Sharp-MAML_low. For example, to run Miniimagenet 5-way 1-shot, run:
```bash
python train_mini-sharp.py /path/to/data --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.1 --batch-size 4 --num-workers 8 --num-epochs 600 --output-folder /path/to/results --num-steps 5 --alpha 0.0005
```
The meta-training script creates a configuration file you can use to meta-test your model. You can use [`test.py`](test.py) to meta-test your model:
```bash
python test.py /path/to/results/config.json
```

### Citation
If you use this code, please cite the following reference:
```
@inproceedings{abbas2022,
 author    = {Momin Abbas and Quan Xiao and Lisha Chen and Pin-Yu Chen and Tianyi Chen},
 title     = {Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning},
 year      = {2022},
 booktitle = {Proceedings of International Conference on Machine Learning},
 address   = {Maryland, MD},
 }
