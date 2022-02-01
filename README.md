# Sharp-MAML
### Platform
* Python: 3.9.7
* Pytorch: 1.11.0

### Model 
Standard baseline 4-layer convolutional NN model comprising of 4 modules with a 3 × 3 convolutions with 64 filters followed by batch normalization, a ReLU non-linearity, and a 2 × 2 max-pooling.

### Sharp-MAML (lower)
To run Sharp-MAML (lower) use:
```bash
python3 train_lower.py data --num-shot 1 --num-ways 20 --download --use-cuda
```
### Sharp-MAML (upper/both)
To run Sharp-MAML (uppper) or Sharp-MAML (both) use: 
```bash
python3 train_both.py data --num-shot 1 --num-ways 20 --download --use-cuda
```
Note: set the argument 'SAM_lower' as 'False' to use only Sharp-MAML_upper

### Save Model
After training, the trained model file is saved in the '/save_results' folder in '.th' file format using the model’s state_dict with the torch.save() function. The saved model can be loaded and used to test the model on the held-out testing dataset using model.load_state_dict(torch.load(PATH)).

# Model-Agnostic Meta-Learning (MAML)

> Chelsea Finn, Pieter Abbeel, Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *International Conference on Machine Learning (ICML)*, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

### Citation

```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```
