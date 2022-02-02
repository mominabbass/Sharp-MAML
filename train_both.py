import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
from collections import OrderedDict
from torchmeta.modules import MetaModule
import time
import math
from maml.datasets import get_benchmark_by_name
from maml.utils import tensors_to_device, compute_accuracy
from torchmeta.datasets.helpers import omniglot
#from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from model import ConvolutionalNeuralNetwork
from utils import get_accuracy
from sam import SAM
from sam_folder.model.smooth_cross_entropy import smooth_crossentropy
from sam_folder.utility.bypass_bn import enable_running_stats, disable_running_stats
from sam_folder.model.wide_res_net import WideResNet
from sam_folder.utility.step_lr import StepLR

logger = logging.getLogger(__name__)

def evaluate(model, dataloader, max_batches=500, verbose=True, **kwargs):
    mean_outer_loss, mean_accuracy, count = 0., 0., 0
    with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
        for results in evaluate_iter(model, dataloader, max_batches=max_batches):
            pbar.update(1)
            count += 1
            mean_outer_loss += (results['mean_outer_loss']
                                - mean_outer_loss) / count
            postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
            if 'accuracies_after' in results:
                mean_accuracy += (np.mean(results['accuracies_after'])
                                  - mean_accuracy) / count
                postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
            pbar.set_postfix(**postfix)

    mean_results = {'mean_outer_loss': mean_outer_loss}
    if 'accuracies_after' in results:
        mean_results['accuracies_after'] = mean_accuracy

    return mean_results

def evaluate_iter(model, dataloader, max_batches=500, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    num_batches = 0
    model.eval()
    while num_batches < max_batches:
        for batch in dataloader:
            if num_batches >= max_batches:
                break

            batch = tensors_to_device(batch, device=device)
            _, results = get_outer_loss(model, batch)
            yield results

            num_batches += 1


def get_outer_loss(model, batch):
    if 'test' not in batch:
        raise RuntimeError('The batch does not contain any test dataset.')

    _, test_targets = batch['test']
    num_tasks = test_targets.size(0)
    is_classification_task = (not test_targets.dtype.is_floating_point)
    results = {
        'num_tasks': num_tasks,
        'inner_losses': np.zeros((1,
            num_tasks), dtype=np.float32),
        'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
        'mean_outer_loss': 0.
    }
    if is_classification_task:
        results.update({
            'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
        })

    mean_outer_loss = torch.tensor(0., device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
            in enumerate(zip(*batch['train'], *batch['test'])):
        params, adaptation_results = adapt(model, train_inputs, train_targets,
            is_classification_task=is_classification_task,
            num_adaptation_steps=1,
            step_size=0.1, first_order=False)
        loss_function=F.cross_entropy
        results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
        if is_classification_task:
            results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

        with torch.set_grad_enabled(model.training):
            test_logits = model(test_inputs, params=params)
            outer_loss = loss_function(test_logits, test_targets)
            results['outer_losses'][task_id] = outer_loss.item()
            mean_outer_loss += outer_loss

        if is_classification_task:
            results['accuracies_after'][task_id] = compute_accuracy(
                test_logits, test_targets)

    mean_outer_loss.div_(num_tasks)
    results['mean_outer_loss'] = mean_outer_loss.item()

    return mean_outer_loss, results


def adapt(model, inputs, targets, is_classification_task=None,
          num_adaptation_steps=1, step_size=0.1, first_order=False):
    if is_classification_task is None:
        is_classification_task = (not targets.dtype.is_floating_point)
    params = None
    loss_function=F.cross_entropy
    results = {'inner_losses': np.zeros(
        (num_adaptation_steps,), dtype=np.float32)}

    for step in range(num_adaptation_steps):
        logits = model(inputs, params=params)
        inner_loss = loss_function(logits, targets)
        results['inner_losses'][step] = inner_loss.item()

        if (step == 0) and is_classification_task:
            results['accuracy_before'] = compute_accuracy(logits, targets)

        model.zero_grad()
        params = gradient_update_parameters(model, inner_loss,
            step_size=step_size, params=params,
            first_order=(not model.training) or first_order)

    return params, results

def gradient_update_parameters_new(model,
                               train_input,
                               train_target,
                               loss,
                               params=None,
                               step_size=0.1,
                               first_order=False, adaptive = False, alpha = 0.0005, sam_lower=True):
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


    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    key_list = params.keys()
    items_list = params.values()

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)
    if sam_lower:
        params_list = list(params.values())
        gradnorm = grad_norm(params_list, grads, adaptive)
        scale = alpha / (gradnorm + 1e-12)

        l = list(range(len(grads)))
        old_p = []
        for i in l:
            old_p.append(torch.zeros_like(params_list[i]))

        for i in l:
            e_w = (torch.pow(params_list[i], 2) if adaptive else 1.0) * grads[i] * scale.to(params_list[i])
            params_list[i] = params_list[i].add(e_w)  # climb to the local maximum "w + e(w)"

        params_new = OrderedDict(zip(key_list, params_list))
        train_logit = model(train_input, params=params_new)
        inner_loss = F.cross_entropy(train_logit, train_target)
        model.zero_grad()
        grads_new = torch.autograd.grad(inner_loss,
                                    params_new.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads_new):
                updated_params[name] = param - step_size * grad

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
    shared_device = params_list[0].device  # put everything on the same device, in case of model parallelism
    l = list(range(len(grads)))
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
    logger.warning('Sharp-MAML training in progress')

    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       seed=0,
                       meta_train=True,
                       download=args.download,
                       )

    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size)

    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

    model = ConvolutionalNeuralNetwork(1, args.num_ways, hidden_size=args.hidden_size, final_layer_size=64)  #for mini-imagenet change the first argument to 3 and final_layer_size to 1600
    model.to(device=args.device)
    model.train()
    base_optimizer = torch.optim.Adam
    meta_optimizer = SAM(model.parameters(), base_optimizer, rho=args.alpha,
                            adaptive=args.adap, lr=1e-3)

    print('alpha: ', args.alpha)
    print('SAM_lower: ', args.SAM_lower)
    loss_acc_time_results = np.zeros((args.num_epochs+1, 2))

    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    start_time = time.time()
    # Training loop
    for epoch in range(args.num_epochs):
        print('\n\n\nEpoch#: ', epoch)
        with tqdm(dataloader, total=args.num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                enable_running_stats(model)
                model.zero_grad()

                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)

                val_inputs, val_targets = batch['test']
                val_inputs = val_inputs.to(device=args.device)
                val_targets = val_targets.to(device=args.device)

                outer_loss = torch.tensor(0., device=args.device)
                outer_loss2 = torch.tensor(0., device=args.device)

                for task_idx, (train_input, train_target, val_input,
                        val_target) in enumerate(zip(train_inputs, train_targets,
                        val_inputs, val_targets)):
                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)
                    #inner_loss = smooth_crossentropy(train_logit, train_target, smoothing=0.000).mean()

                    model.zero_grad()
                    params = gradient_update_parameters_new(model,train_input, train_target, inner_loss, step_size=args.step_size,
                                                        first_order=args.first_order, adaptive = args.adap,
                                                        alpha = args.alpha, sam_lower = args.SAM_lower)

                    val_logit = model(val_input, params=params)
                    outer_loss += F.cross_entropy(val_logit, val_target)

                outer_loss.div_(args.batch_size)
                outer_loss.backward()
                meta_optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                accuracy2 = torch.tensor(0., device=args.device)

                for task_idx, (train_input, train_target, val_input,
                        val_target) in enumerate(zip(train_inputs, train_targets,
                        val_inputs, val_targets)):
                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)
                    model.zero_grad()

                    params = gradient_update_parameters_new(model,train_input, train_target, inner_loss, step_size=args.step_size,
                                                        first_order=args.first_order, adaptive = args.adap, alpha = args.alpha, sam_lower = args.SAM_lower)

                    val_logit = model(val_input, params=params)
                    outer_loss2 += F.cross_entropy(val_logit, val_target)

                    with torch.no_grad():
                        accuracy2 += get_accuracy(val_logit, val_target)

                outer_loss2.div_(args.batch_size)
                accuracy2.div_(args.batch_size)
                outer_loss2.backward()
                meta_optimizer.second_step(zero_grad=True)

                print('outer_val_loss: ', outer_loss2)
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy2.item()))

                if batch_idx % 25 == 0:
                    results = evaluate(model, meta_test_dataloader,
                                        max_batches=args.num_batches,
                                        verbose=args.verbose,
                                        desc=epoch_desc.format(epoch + 1))
                    print('\n\n\n test results: ', results)

                if batch_idx >= args.num_batches:
                    break
        loss_acc_time_results[epoch, 0] = accuracy2.item()
        loss_acc_time_results[epoch, 1] = outer_loss2.item()
    print('Training finished, took {:.2f}s'.format(time.time() - start_time))
    print(loss_acc_time_results)

    if args.SAM_lower:
        file_name = 'results_Sharp_MAML_omniglot_alpha_{}_ways_{}_shots_{}_both.npy'.format(args.alpha, args.num_ways, args.num_shots)
    else:
        file_name = 'results_Sharp_MAML_omniglot_alpha_{}_ways_{}_shots_{}_upper.npy'.format(args.alpha, args.num_ways, args.num_shots)
    file_addr = os.path.join('./save_results', file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)   

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument('folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=1, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=20, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--dataset', type=str,
                        choices=['sinusoid', 'omniglot', 'miniimagenet'], default='omniglot',
                        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--num-shots-test', type=int, default=15,
                        help='Number of test example per class. If negative, same as the number '
                             'of training examples `--num-shots` (default: 15).')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--step-size', type=float, default=0.1, help='Step-size for the gradient step for adaptation (default: 0.1).')
    parser.add_argument('--SAM_lower', type=bool, default=False, help='Apply SAM on inner MAML update')
    parser.add_argument('--alpha', type=float, default=0.0005, help='perturbation radius alpha for SAM')
    parser.add_argument('--adap', type=bool, default=False, help='Apply ASAM (adaptive SAM) on MAML')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output-folder', type=str, default=None, help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--download', action='store_true', help='Download the omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available.')

    args = parser.parse_args()
    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    args.device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print('GPU available: ', torch.cuda.is_available())
    train(args)


