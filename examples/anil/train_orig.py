import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
#from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
import time

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def train(args):
    # Training script identical to MAML, see `examples/maml/train.py`
    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')

    dataset = miniimagenet(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
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

    loss_acc_time_results = np.zeros((args.num_batches+1, 2))

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

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

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

    print('Training finished, took {:.2f}s'.format(time.time() - start_time))
    print(loss_acc_time_results)

    file_name = 'results_ANIL_miniimagenet_5way_1shot_t1.npy'
    file_addr = os.path.join('./save_results', file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)  


    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'anil_miniimagenet_'
            '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))

        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Almost No Inner Loop (ANIL)')

    parser.add_argument('folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.01, help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output-folder', type=str, default=None, help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=1000, help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true', help='Download the miniimagenet dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device("cuda:3" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print('GPU available: ', torch.cuda.is_available())
    train(args)
