 #!/usr/bin/env python 
"""
File: ref.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: May 5, 2019
Description: Train biLSTM model model (with or without Self Attention) 
    and a png stimulus representation. Reference-Game experiment.
"""
import uuid
import json
import matplotlib
import dill
import os
import numpy as np
import revtok
import sys
import time

from experiments.utils import AverageMeter, AccuracyMeter, save_student_checkpoint, set_seeds
from experiments.utils import load_single_task_student_checkpoint as load_student_checkpoint
from models.student.lfl.single_task_student import SingleTaskStudent
from utils.dataloaders.vision.reference_load_dataset import get_data_loader
from utils.constants import Constants

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as data
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dim-l', type=int, default=100,
                        help='hidden dimensions (language) [default: 100]')
    parser.add_argument('--output-dim-l', type=int, default=100,
                        help='output dimensions (language) [default: 100]')
    parser.add_argument('--num-layers-l', type=int, default=1,
                        help='number of layers in RNN[default: 1]')
    parser.add_argument('--dropout-l', type=float, default=0.0,
                        help='dropout probability (language) [default: 0.0]')

    parser.add_argument('--self-att', action='store_true', default=False,
                    help='whether to use self attention [default: False]')
    parser.add_argument('--d-dim-l', type=int, default=100,
                    help='d-dim for self attention [default: 100]')
    parser.add_argument('--r-dim-l', type=int, default=5,
                    help='number of attention hops for self attention [default: 5]')

    parser.add_argument('--input-dim-s', type=int, default=78,
                        help='input representation dimensions (stim) [default: 78]')
    parser.add_argument('--hidden-dim-s', type=int, default=100,
        help='hidden representation dimensions (stim) [default: 100]')
    parser.add_argument('--output-dim-s', type=int, default=100,
        help='output representation dimensions (stim) [default: 100]')
    parser.add_argument('--num-layers-s', type=int, default=3,
                        help='number of layers in stim representation network [default: 3]')
    parser.add_argument('--dropout-s', type=float, default=0.0,
                        help='dropout probability (stim) [default: 0.0]')


    parser.add_argument('--hidden-dim-student', type=int, default=100,
        help='hidden representation dimensions (student) [default: 100]')
    parser.add_argument('--num-layers-student', type=int, default=3,
                        help='number of layers in student prediction network [default: 3]')
    parser.add_argument('--dropout-student', type=float, default=0.0,
                        help='dropout probability (student) [default: 0.0]')

    parser.add_argument('--name', type=str, default='', help='model name')
    parser.add_argument('--train-obj', type=str, default='ground_truth',
                        help='ground_truth, teacher, student [default: ground_truth]')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 32]')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train [default: 10]')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate [default: 3e-4]')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='./data/reference/pilot_coll1/{}/dataset.tsv',
                        help='file template for dataset')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='use batch normalization')
    parser.add_argument('--out-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/reference/models',
                        help='where to save models [default: /mnt/fs5/schopra/ratchet/lfl/student/reference/models]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='utilize pretrained resnet')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Model IDs
    if args.name == '':
        model_id = str(uuid.uuid4())
    else:
        model_id = args.name
    model_ids_dir = os.path.join(args.out_dir, 'model_ids')

    # Make directories
    if not os.path.isdir(model_ids_dir):
        os.makedirs(model_ids_dir)  
    args.out_dir = os.path.join(args.out_dir, 'models')
    if not os.path.isdir(os.path.join(args.out_dir, model_id)):
        os.makedirs(os.path.join(args.out_dir, model_id))

    # Set seeds
    set_seeds()

    # Load Data
    train_vocab_fp = os.path.join(args.out_dir, model_id, 'vocab.pkl')
    val_vocab_fp = os.path.join(args.out_dir, model_id, 'val_vocab.pkl')
    train_loader, train_vocab = get_data_loader('./data/reference/pilot_coll1', 'train', train_vocab_fp, batch_size=args.batch_size)
    val_loader, val_vocab = get_data_loader('./data/reference/pilot_coll1', 'val', val_vocab_fp, batch_size=args.batch_size)

    kwargs = {
        'language_model_type': 'bilstm',
        'stim_model_type': 'resnet',
        'pretrained': args.pretrained,
        'embeddings': 'glove',

        'h_dim_l': args.hidden_dim_l,
        'o_dim_l': args.output_dim_l,
        'd_prob_l': args.dropout_l,
        'num_layers_l': args.num_layers_l,

        'with_self_att': args.self_att,
        'd_dim_l': args.d_dim_l,
        'r_dim_l': args.r_dim_l,

        'i_dim_s': args.input_dim_s,
        'h_dim_s': args.hidden_dim_s,
        'o_dim_s': args.output_dim_s,
        'd_prob_s': args.dropout_s,
        'num_layers_s': args.num_layers_s,

        'hidden_dim_student': args.hidden_dim_student,
        'num_layers_student': args.num_layers_student,
        'd_prob_student': args.dropout_student,
        'output_dim': 1,

        'train_obj': args.train_obj,
        'batch_norm': args.bn,
        'model_id': model_id,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'task': 'ref',
    }

    with open(os.path.join(model_ids_dir, '{}.json'.format(kwargs['model_id'])), 'w') as model_params_f:
        json.dump(kwargs, model_params_f)

    kwargs['concept_vocab_field'] = None
    kwargs['reference_vocab_field'] = train_vocab

    def compute_loss(batch):
        """ Compute loss.
        """
        stims, (texts, text_lengths), y = batch

        if args.cuda:
            stims = stims.to('cuda')
            texts = texts.to('cuda')
            text_lengths = text_lengths.to('cuda')
            y = y.to('cuda')

        batch_size = int(stims.shape[0] / 3)
        logits, alphas = student(texts, stims, text_lengths)
        logits = logits.view(batch_size, 3)
        loss = F.cross_entropy(logits, y)
        return loss, alphas, logits, batch_size


    def compute_self_att_loss(alphas):
        """ Compute self attention loss.
        """ 
        if args.self_att:
            assert(alphas is not None), "Self Attention should have been applied"
            I = torch.eye(args.r_dim_l)
            if args.cuda:
                I = I.to('cuda')
            I = I.repeat(alphas.shape[0], 1, 1)
            alphas_t = torch.transpose(alphas, 1, 2).contiguous()
            return torch.norm(torch.bmm(alphas, alphas_t) - I)
        else:
            return 0.0


    def train(epoch=-1, backprop=True):
        """ Train model for a single epoch.
        """
        # Data loading & progress visualization
        loss_meter = AverageMeter()
        acc_meter = AccuracyMeter()
        pbar = tqdm(total=len(train_loader))
        
        if backprop:
            student.train()
        else:
            student.eval()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, alphas, logits, batch_size = compute_loss(batch)
            loss += compute_self_att_loss(alphas)

            if backprop:
                loss.backward()
                optimizer.step()

            # Update progress
            acc_meter.update(logits, batch_size, refGame=True, cuda=args.cuda)
            loss_meter.update(loss.data.item(), batch_size)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))
            pbar.update()
        pbar.close()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))
        print('Train Accuracies:')
        acc_meter.print(True)
        return loss_meter.avg


    def val():
        """ Run model through validation dataset.
        """
        loss_meter = AverageMeter()
        acc_meter = AccuracyMeter()
        pbar = tqdm(total=len(val_loader))
        student.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, alphas, logits, batch_size = compute_loss(batch)
                loss += compute_self_att_loss(alphas)

                # Update progress
                acc_meter.update(logits, batch_size, refGame=True, cuda=args.cuda)
                loss_meter.update(loss.data.item(), batch_size)
                pbar.update()
        pbar.close()
        print('====> Validation set loss: {:.4f}'.format(loss_meter.avg))
        print('Validation Accuracies:')
        acc_meter.print(True)
        return loss_meter.avg, acc_meter


    # Model Training
    set_seeds()
    student = SingleTaskStudent(**kwargs)
    kwargs.pop('reference_vocab_field', None) # remove vocab_field, as it is not serializable
    if args.cuda:
        student = student.to('cuda')
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    best_acc = 0.0
    best_epoch = -1
    losses = np.zeros((args.epochs, 2))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss, val_acc_meter = val()
        losses[epoch - 1, 0] = train_loss
        losses[epoch - 1, 1] = val_loss

        # keep track of best weights -- this is equivalent
        # to a simple version of early-stopping
        is_best = best_acc < val_acc_meter.compute_s_acc()
        if is_best:
            best_acc = val_acc_meter.compute_s_acc()
            best_epoch = epoch

        # save weights to dict
        save_student_checkpoint(
            {
                'state_dict': student.state_dict(),
                'val_loss': val_loss,
                'vocab_file': train_vocab_fp,
                'optimizer': optimizer.state_dict(),
                'language_model_type': 'bilstm',
                'stim_model_type': 'resnet',
                'kwargs': kwargs
            },
            is_best,
            os.path.join(args.out_dir, kwargs['model_id']), 
            'checkpoint.pth.tar',
            'model_best.pth.tar'
        )

    # plot loss over time
    plt.figure()
    plt.plot(range(args.epochs), losses[:, 0], '-', label='train')
    plt.plot(range(args.epochs), losses[:, 1], '-', label='val')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, model_id, 'loss.png'))

    print("Loading best model from disk ...")
    student = load_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'model_best.pth.tar'), use_cuda=args.cuda, refGame=True)
    train(best_epoch, False)
    val()

