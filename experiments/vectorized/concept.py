 #!/usr/bin/env python 
"""
File: concept.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 21, 2019
Description: Train biLSTM model model (with or without Self Attention) 
    and a feature-driven stimulus representation. Concept-Learning experiment.
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

from models.student.lfl.single_task_student import SingleTaskStudent
from utils.constants import Constants
from utils.dataloaders.vectorized.concept_load_dataset import load_dataset, construct_stim_reps, construct_y, convert_to_elmo_ids
from experiments.utils import AverageMeter, AccuracyMeter, save_student_checkpoint, set_seeds
from experiments.utils import load_single_task_student_checkpoint as load_student_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as data
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden-dim-l', type=int, default=100,
                        help='hidden dimensions (language)')
    parser.add_argument('--output-dim-l', type=int, default=100,
                        help='output dimensions (language)')
    parser.add_argument('--num-layers-l', type=int, default=1,
                        help='number of layers in RNN')
    parser.add_argument('--dropout-l', type=float, default=0.0,
                        help='dropout probability (language)')

    parser.add_argument('--self-att', action='store_true', default=False,
                    help='whether to use self attention')
    parser.add_argument('--d-dim-l', type=int, default=100,
                    help='d-dim for self attention')
    parser.add_argument('--r-dim-l', type=int, default=5,
                    help='number of attention hops for self attention')

    parser.add_argument('--input-dim-s', type=int, default=78,
                        help='input representation dimensions (stim)')
    parser.add_argument('--hidden-dim-s', type=int, default=100,
        help='hidden representation dimensions (stim)')
    parser.add_argument('--output-dim-s', type=int, default=100,
        help='output representation dimensions (stim)')
    parser.add_argument('--num-layers-s', type=int, default=3,
                        help='number of layers in stim representation network')
    parser.add_argument('--dropout-s', type=float, default=0.0,
                        help='dropout probability (stim)')


    parser.add_argument('--hidden-dim-student', type=int, default=100,
        help='hidden representation dimensions (student)')
    parser.add_argument('--num-layers-student', type=int, default=3,
                        help='number of layers in student prediction network')
    parser.add_argument('--dropout-student', type=float, default=0.0,
                        help='dropout probability (student)')

    parser.add_argument('--train-obj', type=str, default='ground_truth',
                        help='ground_truth, teacher, student')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='./data/concept/{}/vectorized/concat_informative_dataset.tsv',
                        help='file template for dataset')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='use batch normalization')
    parser.add_argument('--embeddings', type=str, default='glove',
                        help='embeddings to utilize')
    #  parser.add_argument('--out-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/concept/models',
                        #  help='where to save models')
    # XXX: On the cocolab cluster set this to an /mnt/fsX directory!
    parser.add_argument('--out-dir', type=str, default='./saves/',
                        help='where to save models')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Model IDs
    model_id = str(uuid.uuid4())
    model_ids_dir = os.path.join(args.out_dir, 'model_ids')
    if not os.path.isdir(model_ids_dir):
        os.makedirs(model_ids_dir)  

    # Set seeds
    set_seeds()

    args.out_dir = os.path.join(args.out_dir, 'models')
    if not os.path.isdir(os.path.join(args.out_dir, model_id)):
        os.makedirs(os.path.join(args.out_dir, model_id))

    # Construct Data Loaders &  Iterators
    train_data, val_data, test_data, fields, stim_fields = load_dataset(args.data)
    sort_key = lambda x: len(x.text)
    if args.cuda:
        # GPU available
        train_loader, val_loader = data.Iterator.splits(
                (train_data, val_data), sort_key=sort_key, sort_within_batch=True,
                batch_sizes=(args.batch_size, args.batch_size), device=torch.device('cuda'))
    else:
        # CPU only
        train_loader, val_loader = data.Iterator.splits(
                (train_data, val_data), sort_key=sort_key, sort_within_batch=True,
                batch_sizes=(args.batch_size, args.batch_size), device=torch.device('cpu'))


    # Construct vocabulary objects & write to disk
    if args.embeddings == "glove":
        fields['text'][1].build_vocab(train_data, vectors="glove.840B.300d")
        torch.save(fields['text'][1], os.path.join(args.out_dir, model_id, 'vocab.pkl'), pickle_module=dill)
    elif args.embeddings == "elmo":
        fields['text'][1].build_vocab(train_data)
        torch.save(fields['text'][1], os.path.join(args.out_dir, model_id, 'vocab.pkl'), pickle_module=dill)
        print("Using ELMO's pre-trained embeddings")
    else:
        raise Exception("Invalid Embeddings Type")

    for _, t in fields.items():
        c, field = t
        if c != 'text':
            field.build_vocab(train_data)
    
  
    kwargs = {
        'language_model_type': 'bilstm',
        'stim_model_type': 'featureMLP',

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
        'output_dim': 2,

        'train_obj': args.train_obj,
        'batch_norm': args.bn,
        'embeddings': args.embeddings,
        'model_id': model_id,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'task': 'concept',
    }

    with open(os.path.join(model_ids_dir, '{}.json'.format(kwargs['model_id'])), 'w') as model_params_f:
        json.dump(kwargs, model_params_f)

    kwargs['concept_vocab_field'] = fields['text'][1]
    kwargs['reference_vocab_field'] = None

    def compute_loss(batch):
        """ Compute loss.
        """
        (x_l, x_l_lengths) = batch.text
        if args.embeddings == "elmo":
            x_l_reversed = vocab_field.reverse(x_l.data)
            x_l = convert_to_elmo_ids(x_l_reversed, args.cuda)
            x_l_lengths = None
        x_s = construct_stim_reps(batch, stim_fields)
        if args.cuda:
            x_s = x_s.to('cuda')
        logits, alphas = student(x_l, x_s, x_l_lengths)
        y = construct_y(batch, kwargs['train_obj'])
        if args.cuda:
            y = y.to('cuda')
        loss = F.cross_entropy(logits, y)
        return loss, alphas, logits


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
        train_loader.init_epoch()
        
        if backprop:
            student.train()
        else:
            student.eval()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, alphas, logits = compute_loss(batch)
            loss += compute_self_att_loss(alphas)

            if backprop:
                loss.backward()
                optimizer.step()

            # Update progress
            acc_meter.update(logits, batch)
            loss_meter.update(loss.data.item(), batch.batch_size)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))
            pbar.update()
        pbar.close()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))
        print('Training Objective ({}) Train Accuracies:'.format(kwargs['train_obj']))
        acc_meter.print()
        return loss_meter.avg


    def val():
        """ Run model through validation dataset.
        """
        loss_meter = AverageMeter()
        acc_meter = AccuracyMeter()
        pbar = tqdm(total=len(val_loader))
        val_loader.init_epoch()
        student.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, alphas, logits = compute_loss(batch)
                loss += compute_self_att_loss(alphas)
                acc_meter.update(logits, batch)
                loss_meter.update(loss.data.item(), batch.batch_size)
                pbar.update()
        pbar.close()
        print('====> Validation set loss: {:.4f}'.format(loss_meter.avg))
        print('Training Objective ({}) Validation Accuracies:'.format(kwargs['train_obj']))
        acc_meter.print()
        return loss_meter.avg, acc_meter

    # Model Training
    set_seeds()
    student = SingleTaskStudent(**kwargs)
    kwargs.pop('concept_vocab_field', None) # remove vocab_field, as it is not serializable
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
        is_best = best_acc < val_acc_meter.compute_gt_acc()
        if is_best:
            best_acc = val_acc_meter.compute_gt_acc()
            best_epoch = epoch

        # save weights to dict
        save_student_checkpoint(
            {
                'state_dict': student.state_dict(),
                'val_loss': val_loss,
                'vocab_file': os.path.join(args.out_dir, model_id, 'vocab.pkl'),
                'optimizer': optimizer.state_dict(),
                'language_model_type': 'bilstm',
                'stim_model_type': 'featureMLP',
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
    student = load_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'model_best.pth.tar'), use_cuda=args.cuda)
    train(best_epoch, False)
    val()

