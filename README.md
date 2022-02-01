To run Sharp-MAML_lower use:
python3 train_lower.py data --num-shot 1 --num-ways 20 --download --use-cuda


To run Sharp-MAML_upper or Sharp-MAML_both use: (set SAM_lower as False to use only Sharp-MAML_upper)
python3 train_both.py data --num-shot 1 --num-ways 20 --download --use-cuda



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
