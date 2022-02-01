# Sharp-MAML
### Platform
* Python: 3.9.7
* Pytorch: 1.11.0

### Sharp-MAML (lower)
To run Sharp-MAML$_lower$ use:
```bash
python3 train_lower.py data --num-shot 1 --num-ways 20 --download --use-cuda
```
### Sharp-MAML (upper/both)
To run Sharp-MAML_upper or Sharp-MAML_both use: 
```bash
python3 train_both.py data --num-shot 1 --num-ways 20 --download --use-cuda
```
Note: set the argument 'SAM_lower' as 'False' to use only Sharp-MAML_upper

Note: The code saves the trained model file in the '/save_results' folder in '.th' file format that can be used to test the model on the held-out testing dataset.

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
