 #!/usr/bin/env python 
"""
File: concept_teacher.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 15, 2019
Description: Trains a concept encoder (MLP for vectorized concepts or CNN for
images) and decoder (LSTM).
"""
import ast
import uuid
import json
import matplotlib
import dill
import os
import numpy as np
import revtok
import sys
import time
import copy

from models.teacher.teacher import Teacher
from models.student.lfl.comm_student import Student
from models.comm.comm_game import CommGame
from utils.constants import Constants
from utils.dataloaders.vectorized.load_dataset import load_dataset, construct_y, convert_to_elmo_ids
from experiments.utils import AverageMeter, AccuracyMeter, save_student_checkpoint, set_seeds
from experiments.utils import load_single_task_student_checkpoint as load_student_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as ttdata
import pandas as pd
from tqdm import tqdm

matplotlib.use('agg')
import matplotlib.pyplot as plt

VALID_DATASETS = ['unique_concept', 'concept', 'ref']
VALID_STUDENT_DATASETS = ['concept', 'ref']
DEFAULT_DATAPATHS = dict(unique_concept='./data/concept/{}/vectorized/unique_concept_dataset.tsv',
                         concept='./data/concept/{}/vectorized/concept_dataset.tsv',
                         ref = './data/reference/pilot_coll1/{}/vectorized/ref_dataset.tsv')
DEFAULT_B_SIZES = dict(unique_concept=12, concept=32, ref=32)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden-dim-l-student', type=int, default=100,
                        help='hidden dimensions (language)')
    parser.add_argument('--output-dim-l-student', type=int, default=100,
                        help='output dimensions (language)')
    parser.add_argument('--num-layers-l-student', type=int, default=1,
                        help='number of layers in RNN')
    parser.add_argument('--dropout-l-student', type=float, default=0.0,
                        help='dropout probability (language)')

    parser.add_argument('--self-att', action='store_true', default=False,
                    help='whether to use self attention')
    parser.add_argument('--d-dim-l', type=int, default=100,
                    help='d-dim for self attention')
    parser.add_argument('--r-dim-l', type=int, default=5,
                    help='number of attention hops for self attention')

    parser.add_argument('--input-dim-s-student', type=int, default=78,
                        help='input representation dimensions (stim)')
    parser.add_argument('--hidden-dim-s-student', type=int, default=100,
                        help='hidden representation dimensions (stim)')
    parser.add_argument('--output-dim-s-student', type=int, default=100,
                        help='output representation dimensions (stim)')
    parser.add_argument('--num-layers-s-student', type=int, default=3,
                        help='number of layers in stim representation network')
    parser.add_argument('--dropout-s-student', type=float, default=0.0,
                        help='dropout probability (stim)')


    parser.add_argument('--hidden-dim-student', type=int, default=100,
                        help='hidden representation dimensions (student)')
    parser.add_argument('--num-layers-student', type=int, default=3,
                        help='number of layers in student prediction network')
    parser.add_argument('--dropout-student', type=float, default=0.0,
                        help='dropout probability (student)')

    parser.add_argument('--bn-student', action='store_true', default=False,
                        help='use batch normalization')

    parser.add_argument('--hidden-dim-l-teacher', type=int, default=100,
                        help='hidden dimensions (language)')
    parser.add_argument('--output-dim-l-teacher', type=int, default=100,
                        help='output dimensions (language)')
    parser.add_argument('--num-layers-l-teacher', type=int, default=1,
                        help='number of layers in RNN')
    parser.add_argument('--dropout-l-teacher', type=float, default=0.0,
                        help='dropout probability (language)')

    parser.add_argument('--input-dim-s-teacher', type=int, default=78,
                        help='input representation dimensions (stim)')
    parser.add_argument('--hidden-dim-s-teacher', type=int, default=100,
                        help='hidden representation dimensions (stim)')
    parser.add_argument('--output-dim-s-teacher', type=int, default=100,
                        help='output representation dimensions (stim)')
    parser.add_argument('--num-layers-s-teacher', type=int, default=3,
                        help='number of layers in stim representation network')
    parser.add_argument('--dropout-s-teacher', type=float, default=0.0,
                        help='dropout probability (stim)')
    parser.add_argument('--bn-teacher', action='store_true', default=False,
                        help='use batch normalization')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='temperature for gumbel softmax')

    # Current usage: specify teacher or student datasets, and not comm datasets,
    # if you want to train those models in isolation
    # If specifying comm datasets, use student datasets as pretraining data
    # It is expected to only run one of these types of experiments at a time
    parser.add_argument('--teacher-dsets', type=str, nargs='*', default=[],
                        help='which data to train the teacher on,'
                        ' in which order (ref or concept)')
    parser.add_argument('--student-dsets', type=str, nargs='*', default=[],
                        help='which data to train the student on,'
                        ' in which order (ref or concept);'
                        ' if comm-dsets is specified, this is used as pretraining data')
    parser.add_argument('--comm-dsets', type=str, nargs='*', default=[],
                        help='which data to play the communication game with,'
                        ' in which order (ref or concept)')
    parser.add_argument('--pretrain-teacher', action='store_true', default=False,
                        help='whether or not to pretrain the comm game teacher')
    # This argument is a bit tricky: the idea is that Python can parse strings
    # as Python literals (including dicts) with the ast.literal_eval function
    # TODO finish this
    # NOTE I will have to change the type of data that the student accepts,
    # since I'll be generating it from the teacher data
    # (so at the very least it will be much easier that way)
    parser.add_argument('--datapaths', type=ast.literal_eval, 
                        default=DEFAULT_DATAPATHS,
                        help='paths to data files; input as python dict notation'
                        ' for each dataset specified in --data'
                        ' NOTE: only use single quote marks in the dict')
    parser.add_argument('--indiv-bsizes', type=ast.literal_eval, 
                        default=DEFAULT_B_SIZES, metavar='N',
                        help='batch size for individual (pre)training'
                        ' input in the form of a Python dict: one size for each dataset'
                        ' (see help for the --datasets option)')
    parser.add_argument('--comm-bsizes', type=ast.literal_eval, 
                        default=DEFAULT_B_SIZES, metavar='N',
                        help='batch size for communication training'
                        ' input in the form of a Python dict: one size for each dataset'
                        ' (see help for the --datapaths option)')
    parser.add_argument('--indiv-epochs', type=int, nargs='+', default=[10], metavar='N',
                        help='number of epochs to (pre)train individual models'
                       ' (input one value for each dataset or one value total'
                       ' to be applied to every dataset)')
    parser.add_argument('--comm-epochs', type=int, nargs='+', default=[10], metavar='N',
                        help='number of epochs to train communication model'
                       ' (input one value for each dataset or one value total'
                       ' to be applied to every dataset)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embeddings', type=str, default='glove',
                        help='embeddings to use')
    #  parser.add_argument('--out-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/concept/models',
                        #  help='where to save models')
    # XXX: On the cocolab cluster set this to an /mnt/fsX directory!
    parser.add_argument('--out-dir', type=str, default='./saves/',
                        help='where to save models')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--greedy', action='store_true', default=False,
                        help='enables greedy language sampling')
    parser.add_argument('--lemmatized', type=bool, default=True,
                        help='enables or disables lemmatization for tokenization')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed to use')
    parser.add_argument('--use-best', type=bool, default=True, 
                        help='stop training on each dset early (default True)')
    parser.add_argument('--fix-student', action='store_true',
                        help='fix the student model in the communication game')

    # Argument post-processing
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    assert any([args.teacher_dsets, args.student_dsets, args.comm_dsets]), \
            "Please set either teacher-dsets, student-dsets or comm-dsets."
    assert not(args.teacher_dsets and (args.student_dsets or args.comm_dsets)), \
            "Please run one experiment at a time (use student-dsets \
            to indicate pretraining datasets for the comm game.)"
    # Check validity of datasets argument
    for dset_list in [args.teacher_dsets, args.student_dsets, args.comm_dsets]:
        if dset_list:
            assert all([dset in VALID_DATASETS for dset in dset_list]), \
                    "Unknown dataset in --data option!"

    if args.teacher_dsets or args.student_dsets:
        if len(args.indiv_bsizes) == 1:
            # Hacky python trick: the or below returns the nonempty one
            args.indiv_bsizes *= len(args.teacher_dsets or args.student_dsets)
        if len(args.indiv_epochs) == 1:
            args.indiv_epochs *= len(args.teacher_dsets or args.student_dsets)
    if args.comm_dsets:
        if len(args.comm_bsizes) == 1:
            args.comm_bsizes *= len(args.comm_dsets)
        if len(args.comm_epochs) == 1:
            args.comm_epochs *= len(args.comm_dsets)

    return args

def make_model_dir(out_dir):
    # Model IDs
    ### Models are stored in ./saves/models, in folders with their id as the name
    model_id = str(uuid.uuid4())
    out_dir = os.path.join(out_dir, 'models')
    if not os.path.isdir(os.path.join(out_dir, model_id)):
        os.makedirs(os.path.join(out_dir, model_id))
    return model_id, os.path.join(out_dir, model_id)

def get_loaders(dsets, text_field=None):
    '''
    Params:
        dsets: a list of the names of the datasets to be used in the experiment
        text_field: the torchtext field object to use for the text column of the data
            use a previously generated text_field to ensure the same stoi mappings
    '''
    # Don't build a new vocab if a text field was passed in
    build_vocab = False if text_field else True
    loaders = {}
    train_data = []
    for dset in set(dsets):
        # If text_field is None, then load_dataset creates a text field and returns it
        # Otherwise it just uses the one passed in
        # (If it was previously None, now it will be the one that was created)
        train, val, text_field = load_dataset(args.datapaths[dset], 
                                          lemmatized=args.lemmatized,
                                          text_field=text_field)
        train_data.append(train)
        if args.cuda:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            # XXX use different batch sizes for comm game (i.e. actually use that arg)
        train_loader, val_loader = ttdata.Iterator.splits(
            (train, val), sort_key=lambda x: len(x.text), sort_within_batch=True,
            batch_sizes=(args.indiv_bsizes[dset], args.indiv_bsizes[dset]), device=device)

        loaders[dset] = dict(train=train_loader, val=val_loader)

    # Construct vocabulary objects & write to disk
    ### glove by default
    ### passing in train_data tells torch that the words in train_data['text'] are the words it should use as keys
    ### note: the (gigantic) vocab file is stored in .\.vector_cache
    if args.embeddings == "glove":
        vectors = "glove.840B.300d"
    elif args.embeddings == "elmo":
        # To be vectorized later
        vectors = None
    else:
        raise Exception("Invalid Embeddings Type")
    if build_vocab:
        text_field.build_vocab(*train_data, vectors=vectors)
    # XXX Do we need this?
    # torch.save(text_field, os.path.join(args.out_dir, model_id, dataset+'_vocab.pkl'), pickle_module=dill)
    return loaders, text_field

def make_kwargs(args, seed, model_id, other={}):
    kwargs = {
        'model_id': model_id,
        'stim_model_type': 'featureMLP',
        'seed': seed,
        'cuda': args.cuda,
        'use_best': args.use_best,

        'h_dim_l_student': args.hidden_dim_l_student,
        'o_dim_l_student': args.output_dim_l_student,
        'd_prob_l_student': args.dropout_l_student,
        'num_layers_l_student': args.num_layers_l_student,

        'with_self_att': args.self_att,
        'd_dim_l': args.d_dim_l,
        'r_dim_l': args.r_dim_l,

        'i_dim_s_student': args.input_dim_s_student,
        'h_dim_s_student': args.hidden_dim_s_student,
        'o_dim_s_student': args.output_dim_s_student,
        'd_prob_s_student': args.dropout_s_student,
        'num_layers_s_student': args.num_layers_s_student,

        'h_dim_student': args.hidden_dim_student,
        'num_layers_student': args.num_layers_student,
        'd_prob_student': args.dropout_student,

        'bn_student': args.bn_student,

        'h_dim_l_teacher': args.hidden_dim_l_teacher,
        'o_dim_l_teacher': args.output_dim_l_teacher,
        'd_prob_l_teacher': args.dropout_l_teacher,
        'num_layers_l_teacher': args.num_layers_l_teacher,

        'self_att': args.self_att,
        'd_dim_l': args.d_dim_l,
        'r_dim_l': args.r_dim_l,

        'i_dim_s_teacher': args.input_dim_s_teacher,
        'h_dim_s_teacher': args.hidden_dim_s_teacher,
        'o_dim_s_teacher': args.output_dim_s_teacher,
        'd_prob_s_teacher': args.dropout_s_teacher,
        'num_layers_s_teacher': args.num_layers_s_teacher,
        'bn_teacher': args.bn_teacher,
        'tau': args.tau,

        'pretrain_teacher': args.pretrain_teacher,

        'embeddings': args.embeddings,

        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'teacher_dsets': args.teacher_dsets,
        'student_dsets': args.student_dsets,
        'comm_dsets': args.comm_dsets,
        'indiv_bsizes': args.indiv_bsizes,
        'comm_bsizes': args.comm_bsizes,
        'indiv_epochs': args.indiv_epochs,
        'comm_epochs': args.comm_epochs,
        'lemmatized': args.lemmatized,
        'datapaths': args.datapaths,
        'lr': args.lr,
        'fix_student': args.fix_student,

        **other
    }

    with open(os.path.join(args.out_dir, 'models', model_id, 'params.json'), 
              'w') as params_f:
        # indent: when set to something besides None, enables pretty-printing
        # of json file; the specific integer sets the tab size in num. spaces
        json.dump(kwargs, params_f, indent=2)

    return kwargs

def get_samples_and_prototypes(model, loader):
    ### TODO why are so many words mapping to the unknown character?
    orig_lang, gen_lang, gen_lang_greedy, stims_list, pos_list, neg_list = [], [], [], [], [], []
    for batch in loader:
        stims, labels, language, _ = model.get_inputs(batch)
        orig_lang.extend(model.text_field.reverse(language))
        # Get greedy samples
        gen_ids = model.sample(stims, labels, greedy=True)
        # Return value is (sampled ids, sampled lengths)
        gen_lang_greedy.extend(model.text_field.reverse(gen_ids[0]))
        # Get normal samples
        gen_ids = model.sample(stims, labels, greedy=False)
        gen_lang.extend(model.text_field.reverse(gen_ids[0]))
        stims_list.extend(stims.tolist())
        # Get prototypes (for visualization)
        pos_prototypes, neg_prototypes = model.get_prototypes(stims, labels)
        pos_list.extend(pos_prototypes.tolist())
        neg_list.extend(neg_prototypes.tolist())
    df = pd.DataFrame(
        list(zip(stims_list, orig_lang, gen_lang_greedy, gen_lang, pos_list, neg_list)), 
        columns=['stims', 'orig_lang', 'gen_lang_greedy', 'gen_lang', 'pos_prototypes', 'neg_prototypes'])
    return df
        

def train(model, epoch, train_loader, optimizer, show_acc=False):
    """ Train model for a single epoch.
    """
    ### Model is declared in global space 
    ### (although after this function, which is allowed, I guess?)
    # Data loading & progress visualization
    ### loss_meter stores the loss of the current batch and the average loss
    loss_meter = AverageMeter()
    ### tqdm is a progress bar for iterations of a loop through an iterator
    ### generally you wrap the iterable in it, like tqdm(range), to automatically keep track of #iterations
    ### but you can also use pbar.update() to tell it to iterate manually
    train_loader.init_epoch()
    pbar = tqdm(total=len(train_loader))
    n_correct, n_total = 0, 0
    
    # Sets model in training mode
    model.train()
    ### Enumerate produces a (counter, item) pair for each item in an iterator
    ### "train_loader" is an iterator of minibatches
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss, logits = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        # Update progress
        if show_acc:
            batch_correct, batch_total = model.compute_acc(logits)
            n_correct += batch_correct
            n_total += batch_total
        loss_meter.update(loss.data.item(), batch.batch_size)
        ### Every so often, print progress information (example #, average loss over this epoch, etc.)
        ### also resets pbar onto a new line
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_meter.avg))
        pbar.update()
    pbar.close()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))
    if show_acc:
        acc = n_correct/n_total
        print("Accuracy: {}%".format(acc*100))
    return loss_meter.avg


def val(model, val_loader, show_acc=False):
    """ Run model through validation dataset.
    """
    loss_meter = AverageMeter()
    pbar = tqdm(total=len(val_loader))
    val_loader.init_epoch()
    model.eval()
    n_correct, n_total = 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            loss, logits = model.compute_loss(batch)
            #loss += compute_self_att_loss(alphas)
            if show_acc:
                new_correct, new_total = model.compute_acc(logits)
                n_correct += new_correct
                n_total += new_total
            loss_meter.update(loss.data.item(), batch.batch_size)
            pbar.update()
    pbar.close()
    print('====> Validation set loss: {:.4f}'.format(loss_meter.avg))
    if show_acc:
        acc = n_correct/n_total
        print("Accuracy: {}%".format(acc*100))
    return loss_meter.avg

def train_model(model, loaders, n_epochs, show_acc=False,
                fix_comm_student=False, **kwargs):
    '''
    Trains a model for n_epochs on loaders['train'], evaluating on 
    loaders['val'].
    Note that show_acc expects that the model has compute_acc(logits) defined,
    where it internally stores and compares to the labels from the last batch.
    fix_comm_student is only to be used with a communication game model, 
    in particular when the student model should be fixed.
    '''
    if kwargs['cuda']:
        model = model.to('cuda')
    if fix_comm_student:
        optimizer = optim.Adam(model.teacher.parameters(),lr=kwargs['lr'],
                               weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'],
                               weight_decay=1e-4)

    best_acc = 0.0
    best_epoch = -1
    ### Losses format: epoch1_loss_train, ..., epochn_loss_train
    ###                epoch1_loss_val,   ..., epochn_loss_val
    losses = np.zeros((n_epochs, 2))
    best_loss = 2**63 # arbitrary really large value
    for epoch in range(1, n_epochs + 1):
        train_loader = loaders['train']
        train_loss = train(model, epoch, train_loader, optimizer, 
                           show_acc=show_acc)
        val_loader = loaders['val']
        val_loss = val(model, val_loader, show_acc=show_acc)
        losses[epoch - 1, 0] = train_loss
        losses[epoch - 1, 1] = val_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best = copy.deepcopy(model)

    return losses, best

def plot_losses(losses, dest):
    '''
    Plot losses over time.
    '''
    plt.figure()
    plt.plot(range(len(losses)), losses[:, 0], '-', label='train')
    plt.plot(range(len(losses)), losses[:, 1], '-', label='val')
    plt.tight_layout()
    plt.legend()
    plt.savefig(dest)

def output_samples_and_prototypes(speaker, loaders, dest):
    train_samples = get_samples_and_prototypes(speaker, loaders['train'])
    train_samples.to_csv('_'.join([dest, 'train.csv']), index=False)
    val_samples = get_samples_and_prototypes(speaker, loaders['val'])
    val_samples.to_csv('_'.join([dest, 'val.csv']), index=False)
        

if __name__ == "__main__":
    args = parse_args()
    seed = set_seeds(args.seed)
    model_id, model_dir = make_model_dir(args.out_dir)
    kwargs = make_kwargs(args, seed, model_id)
    if not args.comm_dsets:
        if args.student_dsets:
            loaders, text_field = get_loaders(args.student_dsets)
            student = Student(text_field, **kwargs)
            for dset_num, dset in enumerate(args.student_dsets):
                losses, best = train_model(student, loaders[dset],
                                           args.indiv_epochs[dset_num],
                                           **kwargs, show_acc=True)
                dest = os.path.join(model_dir,
                                    'teacher_{}_loss{}.png'.format(dset, dset_num))
                plot_losses(losses, dest)
        else:
            loaders, text_field = get_loaders(args.teacher_dsets)
            teacher = Teacher(text_field, **kwargs)
            for dset_num, dset in enumerate(args.teacher_dsets):
                losses, best = train_model(teacher, loaders[dset], 
                                           args.indiv_epochs[dset_num], 
                                           **kwargs)
                if args.use_best:
                    teacher = best
                dest = os.path.join(model_dir, 
                                    'teacher_{}_loss_{}.png'.format(dset, dset_num))
                plot_losses(losses, dest)
                dest = os.path.join(model_dir, 
                                    'teacher_{}_samples_{}'.format(dset, dset_num))
                output_samples_and_prototypes(teacher, loaders[dset], dest)

    elif args.comm_dsets:
        # Initialize student, train it on its pretraining dsets (initialize to use all vocab if not pretraining)
        if args.student_dsets:
            vocab_dsets = args.student_dsets
        else:
            vocab_dsets = VALID_STUDENT_DATASETS
        loaders, text_field = get_loaders(vocab_dsets)
        student = Student(text_field, **kwargs)
        for dset_num, dset in enumerate(args.student_dsets):
            losses, best = train_model(student, loaders[dset],
                                       args.indiv_epochs[dset_num], **kwargs,
                                       show_acc=True)
            if args.use_best:
                student = best
            # XXX this is kinda ugly (more like ugly as all get out)
            dest = os.path.join(model_dir, 
                                'student_{}_loss_{}.png'.format(dset, dset_num))
            plot_losses(losses, dest)
        # Initialize teacher and pretrain it on the same datasets
        # (if pretrain_teacher is set to true)
        teacher = Teacher(text_field, **kwargs)
        if args.pretrain_teacher:
            for dset_num, dset in enumerate(args.student_dsets):
                losses, best = train_model(teacher, loaders[dset], 
                                           args.indiv_epochs[dset_num], 
                                           **kwargs)
                if args.use_best:
                    teacher = best
                dest = os.path.join(model_dir, 
                                    'teacher_{}_loss_{}.png'.format(dset, dset_num))
                plot_losses(losses, dest)
                dest = os.path.join(model_dir, 
                                    'teacher_{}_samples_{}'.format(dset, dset_num))
                output_samples_and_prototypes(teacher, loaders[dset], dest)

        # Train communication game
        # TODO don't reload dsets we already have (just change the epoch size)
        loaders, _ = get_loaders(args.comm_dsets, text_field=text_field) 
        comm = CommGame(teacher, student, **kwargs)
        for dset_num, dset in enumerate(args.comm_dsets):
            losses, best = train_model(comm, loaders[dset], 
                                       args.comm_epochs[dset_num], **kwargs,
                                       show_acc=True, 
                                       fix_comm_student=args.fix_student)
            dest = os.path.join(model_dir,
                                'comm_{}_loss_{}.png'.format(dset, dset_num))
            plot_losses(losses, dest)
            dest = os.path.join(model_dir,
                                'comm_{}_samples_{}'.format(dset, dset_num))
            output_samples_and_prototypes(teacher, loaders[dset], dest)
