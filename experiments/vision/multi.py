 #!/usr/bin/env python 
"""
File: multi.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: May 21, 2019
Description: Train biLSTM model model (with or without Self Attention) 
    and a picture-stimulus representation. Jointly learn how to play
    concept learning games and reference games.
"""
import json
import matplotlib
import dill
import os
import numpy as np
import revtok
import sys
import time
import uuid

from models.student.lfl.multi_task_student import MultiTaskStudent as Student
from utils.constants import Constants
from utils.dataloaders.vision.concept_load_dataset import get_data_loader as get_concept_data_loader
from utils.dataloaders.vision.reference_load_dataset import get_data_loader as get_ref_data_loader
from experiments.utils import AverageMeter, AccuracyMeter, save_student_checkpoint, reverse, set_seeds, load_multi_task_student_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as data
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept-data', type=str,
                        default='/home/schopra8/cultural-ratchet-lfl/data/concept/{}/vision/concat_informative_dataset.tsv',
                        help='file template for concept dataset')
    parser.add_argument('--ref-data', type=str, default='/home/schopra8/cultural-ratchet-lfl/data/reference/pilot_coll1/{}/dataset.tsv',
                        help='file template for reference dataset')
    parser.add_argument('--out-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/multi_task',
                        help='where to save models [default: /mnt/fs5/schopra/ratchet/lfl/student/multi_task]')
    parser.add_argument('--concept-train-obj', type=str, default='ground_truth',
                    help='ground_truth, teacher, student [default: ground_truth]')

    parser.add_argument('--hidden-dim-lang', type=int, default=100,
                        help='hidden dimensions (language) [default: 100]')
    parser.add_argument('--output-dim-lang', type=int, default=100,
                        help='output dimensions (language) [default: 100]')
    parser.add_argument('--num-layers-lang', type=int, default=1,
                        help='number of layers in RNN[default: 1]')
    parser.add_argument('--dropout-lang', type=float, default=0.0,
                        help='dropout probability (language) [default: 0.0]')

    parser.add_argument('--self-att', action='store_true', default=False,
                    help='whether to use self attention [default: False]')
    parser.add_argument('--d-dim-lang', type=int, default=100,
                    help='d-dim for self attention [default: 100]')
    parser.add_argument('--r-dim-lang', type=int, default=5,
                    help='number of attention hops for self attention [default: 5]')
    parser.add_argument('--lemmatized', action='store_true', default=False,
                        help='lemmatization')

    parser.add_argument('--output-dim-stim', type=int, default=100,
        help='output representation dimensions (stim) [default: 100]')

    parser.add_argument('--hidden-dim-shared-student', type=int, default=100,
        help='hidden representation dimensions (student) [default: 100]')
    parser.add_argument('--num-layers-shared-student', type=int, default=2,
        help='number of hidden layers for shared student representation network [default: 2]')
    parser.add_argument('--dropout-shared-student', type=float, default=0.0,
                        help='dropout probability shared student representation [default: 0.0]')

    parser.add_argument('--hidden-dim-concept-student', type=int, default=100,
        help='hidden representation dimensions (student) concept learning [default: 100]')
    parser.add_argument('--num-layers-concept-student', type=int, default=1,
        help='number of hidden layers for concept learning portion of the student network [default: 1]')
    parser.add_argument('--dropout-concept-student', type=float, default=0.0,
                        help='dropout probability concept game student representation [default: 0.0]')

    parser.add_argument('--hidden-dim-ref-student', type=int, default=100,
        help='hidden representation dimensions (student) reference game [default: 100]')
    parser.add_argument('--num-layers-ref-student', type=int, default=1,
        help='number of hidden layers for reference game portion of the student network [default: 1]')
    parser.add_argument('--dropout-ref-student', type=float, default=0.0,
                        help='dropout probability reference game student representation [default: 0.0]')

    parser.add_argument('--concept-loss-weight', type=float, default=0.5,
                        help='weight (out of 1.0) assigned to the concept learning loss [default: 0.5]')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 32]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='use batch normalization')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate [default: 3e-4]')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='utilize pretrained resnet')
    parser.add_argument('--name', type=str, default='', help='model name')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Model IDs
    if args.name == '':
        model_id = str(uuid.uuid4())
    else:
        model_id = args.name
    model_ids_dir = os.path.join(args.out_dir, 'model_ids')
    if not os.path.isdir(model_ids_dir):
        os.makedirs(model_ids_dir)  
 
    # Output directory
    kwargs_dir = args.out_dir
    args.out_dir = os.path.join(args.out_dir, 'models')
    if not os.path.isdir(os.path.join(args.out_dir, model_id)):
        os.makedirs(os.path.join(args.out_dir, model_id))

    # Set seeds
    set_seeds()

    # Load Data
    train_concept_vocab_fp = os.path.join(args.out_dir, model_id, 'concept_vocab.pkl')
    val_concept_vocab_fp = os.path.join(args.out_dir, model_id, 'conept_val_vocab.pkl')
    train_concept_loader, train_concept_vocab = get_concept_data_loader('./data/concept/', 'train', train_concept_vocab_fp, batch_size=args.batch_size)
    val_concept_loader, val_concept_vocab = get_concept_data_loader('./data/concept/', 'val', val_concept_vocab_fp, batch_size=args.batch_size) # TODO: Use val_vocab for validation


    # Load Data
    train_ref_vocab_fp = os.path.join(args.out_dir, model_id, 'ref_vocab.pkl')
    val_ref_vocab_fp = os.path.join(args.out_dir, model_id, 'conept_val_vocab.pkl')
    train_ref_loader, train_ref_vocab = get_ref_data_loader('./data/reference/pilot_coll1', 'train', train_ref_vocab_fp, batch_size=args.batch_size)
    val_ref_loader, val_ref_vocab = get_ref_data_loader('./data/reference/pilot_coll1', 'val', val_ref_vocab_fp, batch_size=args.batch_size) # TODO: Use val_vocab for validation


    # Model Parameters
    kwargs = {
        'language_model_type': 'bilstm',
        'stim_model_type': 'resnet',
        'pretrained': args.pretrained,
        'embeddings': 'glove',

        'concept_data': args.concept_data,
        'ref_data': args.ref_data,
        'out_dir': args.out_dir,
        'concept_train_obj': args.concept_train_obj,

        'h_dim_lang': args.hidden_dim_lang,
        'o_dim_lang': args.output_dim_lang,
        'num_layers_lang': args.num_layers_lang,
        'dropout_lang': args.dropout_lang,

        'self_att': args.self_att,
        'd_dim_lang': args.d_dim_lang,
        'r_dim_lang': args.r_dim_lang,
        'lemmatized': args.lemmatized,

        'o_dim_stim': args.output_dim_stim,
        
        'h_dim_shared_student': args.hidden_dim_shared_student,
        'num_layers_shared_student': args.num_layers_shared_student,
        'dropout_shared_student': args.dropout_shared_student,

        'h_dim_concept_student': args.hidden_dim_concept_student,
        'num_layers_concept_student': args.num_layers_concept_student,
        'dropout_concept_student': args.dropout_concept_student,

        'h_dim_ref_student': args.hidden_dim_ref_student,
        'num_layers_ref_student': args.num_layers_ref_student,
        'dropout_ref_student': args.dropout_ref_student,
        
        'concept_loss_weight': args.concept_loss_weight,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'bn': args.bn,
        'lr': args.lr,
        'log_interval': args.log_interval,
        'cuda': args.cuda,
        'model_id': model_id,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
    }

    if kwargs['concept_loss_weight'] == 0.0:
        task = 'ref'
    elif kwargs['concept_loss_weight'] == 1.0:
        task = 'concept'
    else:
        task = 'multi'
    kwargs['task'] = task

    model_ids_dir = os.path.join(kwargs_dir, 'model_ids')
    if not os.path.isdir(model_ids_dir):
        os.makedirs(model_ids_dir)  
    with open(os.path.join(model_ids_dir, '{}.json'.format(model_id)), 'w') as model_params_f:
        json.dump(kwargs, model_params_f)


    def compute_self_att_loss(alphas):
        """ Compute loss term associated with self attention. 
        """
        if args.self_att:
            assert(alphas is not None), "Self Attention should have been applied"
            I = torch.eye(kwargs['r_dim_lang'])
            if kwargs['cuda']:
                I = I.to('cuda')
            I = I.repeat(alphas.shape[0], 1, 1)
            alphas_t = torch.transpose(alphas, 1, 2).contiguous()
            extra_loss = torch.norm(torch.bmm(alphas, alphas_t) - I)
            return extra_loss
        else:
            return 0.0


    def compute_concept_loss(batch):
        """ Compute concept game loss.
        """
        images, (texts, text_lengths), student_labels, teacher_labels, true_labels, data_ids = batch
        if kwargs['train_obj'] == Constants.TRAIN_OBJECTIVES['ground_truth']:
            labels = true_labels
        elif kwargs['train_obj'] == Constants.TRAIN_OBJECTIVES['teacher']:
            labels = teacher_labels
        elif kwargs['train_obj'] == Constants.TRAIN_OBJECTIVES['student']:
            labels = student_labels
        else:
            raise Exception('Invalid training objective')

        if args.cuda:
            images = images.to('cuda')
            texts = texts.to('cuda')
            text_lengths = text_lengths.to('cuda')
            labels = labels.to('cuda')
            true_labels = true_labels.to('cuda')
            teacher_labels = teacher_labels.to('cuda')
            student_labels = student_labels.to('cuda')

        logits, alphas = student(texts, images, text_lengths, use_concept=True)
        loss = F.cross_entropy(logits, labels)
        all_labels = {
            'ground_truth': true_labels,
            'student': student_labels,
            'teacher': teacher_labels
        }
        return loss, alphas, logits, labels, images.shape[0], all_labels, data_ids


    def compute_ref_loss(batch):
        """ Compute reference game loss.
        """
        stims, (texts, text_lengths), y = batch

        if args.cuda:
            stims = stims.to('cuda')
            texts = texts.to('cuda')
            text_lengths = text_lengths.to('cuda')
            y = y.to('cuda')

        batch_size = int(stims.shape[0] / 3)
        logits, alphas = student(texts, stims, text_lengths, use_concept=False)
        logits = logits.view(batch_size, 3)
        loss = F.cross_entropy(logits, y)
        return loss, alphas, logits, batch_size


    def run_epoch(task, concept_loader, ref_loader, train, epoch, backprop):
        """ Run model through 1 epoch.
        """
        if train:
            epoch_type = 'Training'
        else:
            epoch_type = 'Validation'

        # Loss & Accuracy Meters
        concept_loss_meter = AverageMeter()
        concept_acc_meter = AccuracyMeter()
        ref_loss_meter = AverageMeter()
        ref_acc_meter = AccuracyMeter()
        loss_meter = AverageMeter()

        # Data loader init + visualization
        if task == 'ref':
            pbar = tqdm(total=len(ref_loader))
        elif task == 'concept':
            pbar = tqdm(total=len(concept_loader))
        else:
            pbar = tqdm(total=len(ref_loader) + len(concept_loader))

        # Set model mode
        if backprop:
            student.train()
        else:
            student.eval()

        # Train model on two objectives
        ref_batch_idx = -1
        ref_iterator = iter(ref_loader)
        ref_batches_exist = True
        for concept_batch_idx, concept_batch in enumerate(concept_loader):
            try:
                ref_batch = next(ref_iterator)
                ref_batch_idx += 1
            except StopIteration:
                ref_batches_exist = False
                if task == 'ref':
                    break

            # Compute loss + backprop
            optimizer.zero_grad()
            if task == 'ref':
                ref_loss, ref_alphas, ref_logits, ref_batch_size = compute_ref_loss(ref_batch)
                loss = ref_loss + compute_self_att_loss(ref_alphas)
            elif task == 'concept':
                concept_loss, concept_alphas, concept_logits, concept_labels, concept_batch_size, concept_all_labels, _ = compute_concept_loss(concept_batch) 
                loss = concept_loss + compute_self_att_loss(concept_alphas)
            else:
                ref_loss, ref_alphas, ref_logits, ref_batch_size = compute_ref_loss(ref_batch)
                loss, concept_alphas, concept_logits, concept_labels, concept_batch_size, concept_all_labels, _ = compute_concept_loss(concept_batch) 
                loss = concept_loss * kwargs['concept_loss_weight'] + ref_loss * (1.0 - kwargs['concept_loss_weight'])
                loss += compute_self_att_loss(concept_alphas) + compute_self_att_loss(ref_alphas)

            if backprop:
                loss.backward()
                optimizer.step()

            # Update concept learning progress
            if task == 'concept' or task == 'multi':
                concept_acc_meter.update(concept_logits, concept_batch, refGame=False, cuda=kwargs['cuda'])
                concept_loss_meter.update(concept_loss.data.item(), concept_batch_size)
                if concept_batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tConcept Loss: {:.6f}'.format(
                        epoch, concept_batch_idx * concept_batch_size, len(concept_loader),
                        100. * concept_batch_idx / len(concept_loader), concept_loss_meter.avg))
                pbar.update()

            # Reference game learning progress
            if ref_batches_exist and (task == 'ref' or task == 'multi'):
                ref_acc_meter.update(ref_logits, ref_batch, refGame=True, cuda=kwargs['cuda'])
                ref_loss_meter.update(ref_loss.data.item(), ref_batch_size)
                if ref_batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReference Loss: {:.6f}'.format(
                        epoch, ref_batch_idx * ref_batch_size, len(ref_loader),
                        100. * ref_batch_idx / len(ref_loader), ref_loss_meter.avg))
                pbar.update()

            # Joint learning progress\
            if task == 'multi':
                if ref_batches_exist:
                    loss_meter.update(loss.data.item(), ref_batch_size + concept_batch_size)
                else:
                    loss_meter.update(loss.data.item(), concept_batch_size)
                if (ref_batch_idx + concept_batch_idx) % args.log_interval == 0:
                    print('Train Epoch: {}\tJoint Loss: {:.6f}'.format(
                        epoch, loss_meter.avg))

        
        # Train model on reference objective, if there outstanding batches
        if task == 'ref' or task == 'multi':
            while (ref_batches_exist):
                try:
                    ref_batch = next(ref_iterator)
                    ref_batch_idx += 1
                except StopIteration:
                    break

                # Compute loss + backprop
                optimizer.zero_grad()
                ref_loss, ref_alphas, ref_logits, ref_batch_size = compute_ref_loss(ref_batch)
                loss += ref_loss * (1.0 - kwargs['concept_loss_weight']) + compute_self_att_loss(ref_alphas)
                if backprop:
                    loss.backward()
                    optimizer.step()

                # Reference game learning progress
                ref_acc_meter.update(ref_logits, ref_batch, refGame=True, cuda=kwargs['cuda'])
                ref_loss_meter.update(ref_loss.data.item(), ref_batch_size)
                if ref_batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReference Loss: {:.6f}'.format(
                        epoch, ref_batch_idx * ref_batch_size, len(ref_loader),
                        100. * ref_batch_idx / len(ref_loader), ref_loss_meter.avg))
                pbar.update()
        
        pbar.close()
        if task == 'concept' or task == 'multi':
            print('====> Epoch: {} Average Concept loss: {:.4f}'.format(epoch, concept_loss_meter.avg))
        if task == 'ref' or task == 'multi':
            print('====> Epoch: {} Average Reference loss: {:.4f}'.format(epoch, ref_loss_meter.avg))
        if task == 'multi':
            print('====> Epoch: {} Average Joint loss: {:.4f}'.format(epoch, loss_meter.avg))
        if task != 'ref':
            print('Concept Learning Objective ({}) Concept {} Accuracies:'.format(epoch_type, kwargs['concept_train_obj']))
            concept_acc_meter.print()
        if task != 'concept':
            print('Reference {} Accuracies:'.format(epoch_type))
            ref_acc_meter.print(ground_truth_only=True)
        return loss_meter.avg, concept_loss_meter.avg, ref_loss_meter.avg, concept_acc_meter, ref_acc_meter


    def train(model, task, epoch=-1, backprop=True):
        """ Train model for a single epoch.
        """
        model.languageModel.replace_embeddings(train_concept_vocab, train_ref_vocab)
        if kwargs['cuda']:
            model = model.to('cuda')
        return run_epoch(task, train_concept_loader, train_ref_loader, True, epoch, backprop)

    def val(model, task, epoch):
        """ Run model through validation dataset.
        """
        model.languageModel.replace_embeddings(val_concept_vocab, val_ref_vocab)
        if kwargs['cuda']:
            model = model.to('cuda')
        return run_epoch(task, val_concept_loader, val_ref_loader, False, epoch, False)


    # Model Training
    set_seeds()
    kwargs['concept_vocab_field'] = train_concept_vocab
    kwargs['reference_vocab_field'] = train_ref_vocab
    student = Student(**kwargs)
    kwargs.pop('concept_vocab_field', None)
    kwargs.pop('reference_vocab_field', None)
    if kwargs['cuda']:
        student = student.to('cuda')
    optimizer = optim.Adam(student.parameters(), lr=kwargs['lr'], weight_decay=1e-4)

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    import seaborn as sns
    sns.set_style('whitegrid')

    best_concept_acc = 0.0
    best_ref_acc = 0.0
    best_loss = sys.maxsize

    # 0 -> train joint
    # 1 -> val joint
    # 2 -> train concept
    # 3 -> val concept
    # 4 -> train ref
    # 4 -> val ref
    losses = np.zeros((args.epochs, 6))  

    best_concept_epoch = -1
    best_ref_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_concept_loss, train_ref_loss, _, _ = train(student, task, epoch)
        val_loss, val_concept_loss, val_ref_loss, concept_acc_meter, ref_acc_meter = val(task, epoch)
        losses[epoch - 1, 0] = train_loss
        losses[epoch - 1, 1] = val_loss
        losses[epoch - 1, 2] = train_concept_loss
        losses[epoch - 1, 3] = val_concept_loss
        losses[epoch - 1, 4] = train_ref_loss
        losses[epoch - 1, 5] = val_ref_loss

        # keep track of best weights -- this is equivalent
        # to a simple version of early-stopping
        best_loss = min(val_loss, best_loss)
        if task == 'concept' or task == 'multi':
            is_best_concept = best_concept_acc < concept_acc_meter.compute_gt_acc()
            if is_best_concept:
                best_concept_epoch = epoch
                best_concept_acc = concept_acc_meter.compute_gt_acc()
        if task == 'ref' or task == 'multi':
            is_best_ref = best_ref_acc < ref_acc_meter.compute_gt_acc()
            if is_best_ref:
                best_ref_epoch = epoch
                best_ref_acc = ref_acc_meter.compute_gt_acc()

        # save weights to dict
        if task == 'concept' or task == 'multi':
            save_student_checkpoint(
                {
                    'state_dict': student.state_dict(),
                    'val_loss': val_loss,
                    'concept_vocab_file': os.path.join(args.out_dir, model_id, 'concept_vocab.pkl'),
                    'ref_vocab_file': os.path.join(args.out_dir, model_id, 'ref_vocab.pkl'),
                    'optimizer': optimizer.state_dict(),
                    'language_model_type': 'bilstm',
                    'stim_model_type': 'featureMLP',
                    'kwargs': kwargs
                },
                is_best_concept,
                os.path.join(args.out_dir, model_id), 
                'checkpoint.pth.tar',
                'concept_model_best.pth.tar'
            )
        if task == 'ref' or task == 'multi':
            save_student_checkpoint(
                {
                    'state_dict': student.state_dict(),
                    'val_loss': val_loss,
                    'concept_vocab_file': os.path.join(args.out_dir, model_id, 'concept_vocab.pkl'),
                    'ref_vocab_file': os.path.join(args.out_dir, model_id, 'ref_vocab.pkl'),
                    'optimizer': optimizer.state_dict(),
                    'language_model_type': 'bilstm',
                    'stim_model_type': 'featureMLP',
                    'kwargs': kwargs
                },
                is_best_ref,
                os.path.join(args.out_dir, model_id), 
                'checkpoint.pth.tar',
                'ref_model_best.pth.tar'
            )

    # plot loss over time
    plt.figure()
    plt.plot(range(args.epochs), losses[:, 0], '-', label='train')
    plt.plot(range(args.epochs), losses[:, 1], '-', label='val')
    if task == 'multi':
        plt.plot(range(args.epochs), losses[:, 2], '-', label='train_concept')
        plt.plot(range(args.epochs), losses[:, 3], '-', label='val_concept')       
        plt.plot(range(args.epochs), losses[:, 4], '-', label='train_ref')
        plt.plot(range(args.epochs), losses[:, 5], '-', label='val_ref')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, model_id, 'loss.png'))

    if task == 'concept' or task == 'multi':
        print("Loading best model from disk ...")
        print("Concept Learning epoch: {}".format(best_concept_epoch))
        student = load_multi_task_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'concept_model_best.pth.tar'), use_cuda=kwargs['cuda'])
        train(student, task, best_concept_epoch, False)
        val(student, task, best_concept_epoch)

    if task == 'ref' or task == 'multi':
        print("Loading best model from disk ...")
        print("Reference Game epoch: {}".format(best_ref_epoch))
        student = load_multi_task_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'ref_model_best.pth.tar'), use_cuda=kwargs['cuda'])
        train(student, task, best_ref_epoch, False)
        val(student, task, best_ref_epoch)

