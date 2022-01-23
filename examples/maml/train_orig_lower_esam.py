import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
#from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.modules import MetaModule
from collections import OrderedDict
import time
import random
#from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def gradient_update_parameters_new(model,
                               train_input,
                               train_target,
                               loss,
                               l_before,
                               loss_fct,
                               params=None,
                               step_size=0.5,
                               first_order=False, adaptive = False, rho = 0.05, sam_lower=False, beta=1.0, gamma=1.0):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.

    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    # a = torch.ones(len(params)) * beta
    # mask = torch.bernoulli(a)

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    key_list = params.keys()
    items_list = params.values()

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    if sam_lower:
        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True

        params_list = list(params.values())

        ###start new code
        gradnorm = grad_norm(params_list, grads, adaptive)

        scale = rho / (gradnorm + 1e-12) / beta

        l = list(range(len(grads)))

        for i in l:
            e_w = (torch.pow(params_list[i], 2) if adaptive else 1.0) * grads[i] * scale.to(params_list[i])
            # print('e_2w: ', e_w[0:2])
            # print('val1: ', params_list[i][0:2])
            params_list[i] = params_list[i].add(e_w)  # climb to the local maximum "w + e(w)"


        params_new = OrderedDict(zip(key_list, params_list))
        # print('new params', list(params_new.keys())[0])
        # print('old params', list(params.keys())[0])



        with torch.no_grad():
            train_logit = model(train_input, params=params_new)
            l_after = loss_fct(train_logit,  train_target)
            instance_sharpness = l_after-l_before

            # codes for sorting
            prob = gamma
            if prob >= 0.99:
                indices = range(len(train_target))
            else:
                position = int(len(train_target) * prob)
                cutoff, _ = torch.topk(instance_sharpness, position)
                cutoff = cutoff[-1]

                # cutoff = 0
                # select top k%

                indices = [instance_sharpness > cutoff]

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False

        train_logit = model(train_input[indices], params=params_new)
        inner_loss = F.cross_entropy(train_logit, train_target[indices])

        model.zero_grad()
        grads_new = torch.autograd.grad(inner_loss,
                                    params_new.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads_new):
                if random.random() > beta:
                    updated_params[name] = param - step_size[name] * grad
                else:
                    updated_params[name] = param
        else:
            for (name, param), grad in zip(params.items(), grads_new):
                if random.random() > beta:
                    updated_params[name] = param - step_size * grad
                else:
                    updated_params[name] = param
    ###end new code

    else:
        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size * grad

    return updated_params

def grad_norm(params_list, grads, adaptive):
    #print('momin3', len(list(params.values())))import torch

    shared_device = params_list[0].device  # put everything on the same device, in case of model parallelism
    l = list(range(len(grads)))
    #print('l: ', l)
    norm = torch.norm(
                torch.stack([
                    ((torch.abs(params_list[i]) if adaptive else 1.0) * grads[i]).norm(p=2).to(shared_device)
                     for i in l
                    if grads is not None
                ]),
                p=2
           )
    return norm

def train(args):
    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       seed=0,
                       meta_train=True,
                       download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(3,
                                       args.num_ways,
                                       hidden_size=args.hidden_size, 
                                       final_layer_size=1600)
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('rho: ', args.rho)
    print('beta: ', args.beta)
    print('gamma: ', args.gamma)
    print('SAM_lower: ', args.SAM_lower)
    print('Adaptive: ', args.adap)

    loss_acc_time_results = np.zeros((args.num_batches+1, 2))

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Training loop
    start_time = time.time()
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
           
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):
                train_logit = model(train_input)

                inner_loss = F.cross_entropy(train_logit, train_target)
                l_before = loss_fct(train_logit, train_target)

                model.zero_grad()
                # params = gradient_update_parameters(model,
                #                                     inner_loss,
                #                                     step_size=args.step_size,
                #                                     first_order=args.first_order)
                params = gradient_update_parameters_new(model,train_input, train_target, 
                                                        inner_loss, l_before, loss_fct, step_size=args.step_size,
                                                        first_order=args.first_order, adaptive = args.adap,
                                                         rho = args.rho, sam_lower = args.SAM_lower, beta=args.beta, gamma=args.gamma)


                
                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)


            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)

            outer_loss.backward()
            meta_optimizer.step()

            print('test_loss: ', outer_loss)


            loss_acc_time_results[batch_idx, 0] = accuracy.item()
            loss_acc_time_results[batch_idx, 1] = outer_loss.item()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break

    print('ESAM Training finished, took {:.2f}s'.format(time.time() - start_time))
    print(loss_acc_time_results)

    file_name = 'results_BiESAM_{}_beta{}_gamma{}_miniimagenet_5way_1shot_lower_t25.npy'.format(args.rho, args.beta, args.gamma)
    file_addr = os.path.join('./save_results_min', file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)   

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_miniimagenet_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.01, help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--SAM_lower', type=bool, default=True, help='Apply SAM on inner MAML update')
    parser.add_argument('--rho', type=float, default=0.0005, help='radius rho for SAM')
    parser.add_argument('--adap', type=bool, default=False, help='Apply ASAM (adaptive SAM) on MAML')
    parser.add_argument('--beta', type=float, default=0.3, help='Droupout rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Droupout rate')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output-folder', type=str, default=None, help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=1000, help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true', help='Download the miniimagenet dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available.')
    parser.add_argument('--trial', type=int, default=0, help='trial for different run')

    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.trial) if args.use_cuda and torch.cuda.is_available() else "cpu")
    print('GPU available: ', torch.cuda.is_available())
    train(args)