# Sharp-MAML
### Platform
* Python: 3.9.7
* Pytorch: 1.11.0

### Model 
Standard baseline 4-layer convolutional NN model comprising of 4 modules with a 3 × 3 convolutions with 64 filters followed by batch normalization, a ReLU non-linearity, and a 2 × 2 max-pooling.

### Sharp-MAML (lower)
To run Sharp-MAML (lower) use:
```bash
python3 train_lower.py /path/to/data --num-shots 1 --num-ways 5 --download --use-cuda
```
### Sharp-MAML (upper)
To run Sharp-MAML (upper) use: 
```bash
python3 train_both.py /path/to/data --num-shots 1 --num-ways 5 --download --use-cuda
```
Note: In the train_both.py file, set the argument 'SAM_lower' as 'False' to use only Sharp-MAML (upper)

### Sharp-MAML (both)
To run Sharp-MAML (both) use: 
```bash
python3 train_both.py /path/to/data --num-shots 1 --num-ways 5 --download --use-cuda
```
Note: In the train_both.py file, set the argument 'SAM_lower' as 'True' to use Sharp-MAML (both)

### Save Model
After training, the trained model file is saved in the '/save_results' folder in '.th' file format using the model’s state_dict with the torch.save() function. The saved model can be loaded and used to test the model on the held-out testing dataset using model.load_state_dict(torch.load(PATH)).

# Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning

> Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, Tianyi Chen. Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning. [[ArXiv](https://)]

### Citation
If you use this code, please cite the following reference:
```
@inproceedings{abbas2022,
 author    = {Momin Abbas and Quan Xiao and Lisha Chen and Pin-Yu Chen and Tianyi Chen},
 title     = {Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning},
 year      = {2022},
 booktitle     = {Proceedings of International Conference on Machine Learning},
 address   = {Maryland, MD},
 }
```
