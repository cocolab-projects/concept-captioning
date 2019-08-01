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

from models.teacher.teacher import Teacher
from models.student.lfl.comm_student import Student
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
DEFAULT_TEACHER_DATAPATHS = dict(unique_concept='./data/concept/{}/vectorized/unique_concept_dataset.tsv',
                         concept='./data/concept/{}/vectorized/concept_dataset.tsv',
                         ref = './data/reference/pilot_coll1/{}/vectorized/ref_dataset.tsv')
DEFAULT_STUDENT_DATAPATHS = dict(unique_concept='./data/concept/{}/vectorized/unique_concept_dataset.tsv',
                         concept='./data/concept/{}/vectorized/concept_dataset.tsv',
                         ref = './data/reference/pilot_coll1/{}/vectorized/ref_dataset.tsv')
DEFAULT_B_SIZES = dict(unique_concept=12, concept=32, ref=32)


def parse_args():
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

    # Current usage: specify teacher or student datasets, and not comm datasets,
    # if you want to train those models in isolation
    # If specifying comm datasets, use student datasets as pretraining data
    # It is expected to only run one of these types of experiments at a time
    parser.add_argument('--teacher-dsets', type=str, nargs='*', default=None,
                        help='which data to train the teacher on,'
                        ' in which order (ref or concept)')
    parser.add_argument('--student-dsets', type=str, nargs='*', default=None,
                        help='which data to train the student on,'
                        ' in which order (ref or concept);'
                        ' if comm-dsets is specified, this is used as pretraining data')
    parser.add_argument('--comm-dsets', type=str, nargs='*', default=None,
                        help='which data to play the communication game with,'
                        ' in which order (ref or concept)')
    parser.add_argument('pretrain-teacher', action='store_true', default=False,
                        help='whether or not to pretrain the comm game teacher')
    # This argument is a bit tricky: the idea is that Python can parse strings
    # as Python literals (including dicts) with the ast.literal_eval function
    parser.add_argument('--datapaths-student', type=ast.literal_eval, 
                        default=DEFAULT_STUDENT_DATAPATHS,
                        help='paths to data files; input as python dict notation'
                        'for each dataset specified in --data'
                        'NOTE: only use single quote marks in the dict')
    # TODO finish this
    # NOTE I will have to change the type of data that the student accepts,
    # since I'll be generating it from the teacher data
    # (so at the very least it will be much easier that way)
    parser.add_argument('--datapaths-teacher', type=ast.literal_eval, 
                        default=DEFAULT_TEACHER_DATAPATHS,
                        help='paths to data files; input as python dict notation'
                        ' for each dataset specified in --data'
                        ' NOTE: only use single quote marks in the dict'
                        ' Also used as the communication game datapaths')
    parser.add_argument('--batch-sizes', type=ast.literal_eval, 
                        default=DEFAULT_B_SIZES, metavar='N',
                        help='input batch size for training'
                        ' input in the form of a Python dict: one size for each dataset'
                        ' (see help for the --datasets option)')
    parser.add_argument('--epochs', type=int, nargs='+', default=[10], metavar='N',
                        help='number of epochs to train'
                       ' (input one value for each dataset or one value total'
                       ' to be applied to every dataset)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='use batch normalization')
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

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Argument post-processing
    if len(args.batch_sizes) == 1:
        args.batch_sizes = args.batch_sizes * len(args.data)
    if len(args.epochs) == 1:
        args.epochs = args.epochs * len(args.data)

    assert any(args.teacher_data, args.student_data, args.comm_data), \
            "Please set either teacher-data, student-data or comm-data."
    assert (not args.teacher_data and args.comm_data), "Please use student-data \
            to indicate pretraining datasets for the comm game."

    return args

def make_model_dir(out_dir):
    # Model IDs
    ### Models are stored in ./saves/models, in folders with their id as the name
    model_id = str(uuid.uuid4())
    out_dir = os.path.join(out_dir, 'models')
    if not os.path.isdir(os.path.join(out_dir, model_id)):
        os.makedirs(os.path.join(out_dir, model_id))

    # Check validity of datasets argument
    assert all([dataset in VALID_DATASETS for dataset in args.data]), \
            "Unknown dataset in --data option!"
    # Construct Data Loaders &  Iterators
    ### Default args.data is /data/concept/{}/vectorized/concat_informative_dataset
    ### The actual '{}' folders are (raw; maybe not used here), test, train and val
    ### {}_data is a TabularDataset (torchtext.data object)

def get_loaders(dsets, text_field=None):
    '''
    Params:
        dsets: a list of the names of the datasets to be used in the experiment
        text_field: the torchtext field object to use for the text column of the data
            use a previously generated text_field to ensure the same stoi mappings
    '''
    loaders = {}
    train_data = []
    for dset in set(dsets):
        # If text_field is None, then load_dataset creates a text field and returns it
        # Otherwise it just uses the one passed in
        # (If it was previously None, now it will be the one that was created)
        train, val, text_field = load_dataset(args.datapaths[dataset], 
                                          lemmatized=args.lemmatized,
                                          text_field=text_field)
        train_data.append(train)
        if args.cuda:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        train_loader, val_loader = ttdata.Iterator.splits(
            (train, val), sort_key=lambda x: len(x.text), sort_within_batch=True,
            batch_sizes=(args.batch_sizes[dataset], args.batch_sizes[dataset]), device=device)

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
    text_field.build_vocab(*train_data, vectors=vectors)
    torch.save(text_field, os.path.join(args.out_dir, model_id, dataset+'_vocab.pkl'), pickle_module=dill)
    return loaders, text_field

def make_kwargs(args, text_field, other):
    kwargs = {
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

        'batch_norm': args.bn,
        'embeddings': args.embeddings,
        'model_id': model_id,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'data': args.data,
        'batch_sizes': args.batch_sizes,
        'epochs': args.epochs,
        'lemmatized': args.lemmatized,
        'datapaths': args.datapaths,

        # XXX not that ugly, but note that fields is still from the last
        # dataset in the for loop above
        'unk_index': text_field.vocab.stoi[Constants.UNK_TOKEN],
        'pad_index': text_field.vocab.stoi[Constants.PAD_TOKEN],
        'start_index': text_field.vocab.stoi[Constants.START_TOKEN],
        'end_index': text_field.vocab.stoi[Constants.END_TOKEN],

        **other
    }

    # Reality checking for vocab indices
    ### TODO make this cleaner (or figure out how to set these indices manually?)
    ### TODO assert that vocab vectors tensors are equal
    '''
    assert all([dset['fields']['text'][1].vocab.itos == fields['text'][1].vocab.itos for dset in data.values()])
    assert all([dset['fields']['text'][1].vocab.stoi[Constants.UNK_TOKEN] == \
                kwargs['unk_index'] for dset in data.values()])
    assert all([dset['fields']['text'][1].vocab.stoi[Constants.PAD_TOKEN] == \
                kwargs['pad_index'] for dset in data.values()])
    assert all([dset['fields']['text'][1].vocab.stoi[Constants.START_TOKEN] == \
                kwargs['start_index'] for dset in data.values()])
    assert all([dset['fields']['text'][1].vocab.stoi[Constants.END_TOKEN] == \
                kwargs['end_index'] for dset in data.values()])
    '''

    with open(os.path.join(model_ids_dir, '{}.json'.format(kwargs['model_id'])), 'w') as id_f, \
            open(os.path.join(args.out_dir, model_id, 'params.json'), 'w') as params_f:
        # indent: when set to something besides None, enables pretty-printing
        # of json file; the specific integer sets the tab size in num. spaces
        json.dump(kwargs, params_f, indent=2)
        json.dump(kwargs, id_f, indent=2)

    # (placing this code after the dump since the field isn't valid JSON)
    kwargs['vocab_field'] = text_field
    return kwargs

def get_samples_and_prototypes(model, loader):
    ### TODO why are so many words mapping to the unknown character?
    orig_lang, gen_lang, gen_lang_greedy, stims_list, pos_list, neg_list = [], [], [], [], [], []
    for batch in loader:
        stims, labels, language, _ = get_inputs(batch)
        orig_lang.extend(model.text_field.reverse(language))
        gen_ids = teacher.sample(stims, labels, **kwargs)
        gen_lang_greedy.extend(model.text_field.reverse(gen_ids[0]))
        kwargs['greedy_sampling'] = False
        gen_ids = teacher.sample(stims, labels, **kwargs)
        gen_lang.extend(model.text_field.reverse(gen_ids[0]))
        kwargs['greedy_sampling'] = True
        stims_list.extend(stims.tolist())
        pos_prototypes, neg_prototypes = teacher.get_prototypes(stims, labels)
        pos_list.extend(pos_prototypes.tolist())
        neg_list.extend(neg_prototypes.tolist())
    df = pd.DataFrame(
        list(zip(stims_list, orig_lang, gen_lang_greedy, gen_lang, pos_list, neg_list)), 
        columns=['stims', 'orig_lang', 'gen_lang_greedy', 'gen_lang', 'pos_prototypes', 'neg_prototypes'])
    return df
        

def train(model, epoch, train_loader):
    """ Train model for a single epoch.
    """
    ### Model is declared in global space 
    ### (although after this function, which is allowed, I guess?)
    # Data loading & progress visualization
    ### loss_meter stores the loss of the current batch and the average loss
    loss_meter = AverageMeter()
    #acc_meter = AccuracyMeter()
    ### tqdm is a progress bar for iterations of a loop through an iterator
    ### generally you wrap the iterable in it, like tqdm(range), to automatically keep track of #iterations
    ### but you can also use pbar.update() to tell it to iterate manually
    train_loader.init_epoch()
    pbar = tqdm(total=len(train_loader))
    
    # Sets model in training mode
    model.train()
    ### Enumerate produces a (counter, item) pair for each item in an iterator
    ### "train_loader" is an iterator of minibatches
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss, logits = model.compute_loss(batch)

        if backprop:
            loss.backward()
            optimizer.step()

        # Update progress
        #acc_meter.update(logits, batch)
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
    #acc_meter.print()
    return loss_meter.avg


def val(model, val_loader):
    """ Run model through validation dataset.
    """
    loss_meter = AverageMeter()
    #acc_meter = AccuracyMeter()
    pbar = tqdm(total=len(val_loader))
    val_loader.init_epoch()
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            loss, logits = model.compute_loss(batch)
            #loss += compute_self_att_loss(alphas)
            #acc_meter.update(logits, batch)
            loss_meter.update(loss.data.item(), batch.batch_size)
            pbar.update()
    pbar.close()
    print('====> Validation set loss: {:.4f}'.format(loss_meter.avg))
    #acc_meter.print()
    return loss_meter.avg#, acc_meter

def train_model(model, data, n_epochs, **kwargs):
    if kwargs['cuda']:
        model = model.to('cuda')

    for dset_num, dset in enumerate(args.data):
        ### Adam is an optimization algorithm, like SGD, except that it changes its learning rate dynamically
        ### based on recent gradients (low gradients --> high LR, vice versa)
        ### It's similar in this way to root mean square propagation (RMSProp), just a little more sophisticated
        ### People apparently just use it because it works well
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_acc = 0.0
        best_epoch = -1
        ### Losses format: epoch1_loss_train, ..., epochn_loss_train
        ###                epoch1_loss_val,   ..., epochn_loss_val
        losses = np.zeros((args.epochs[dset_num], 2))
        for epoch in range(1, args.epochs[dset_num] + 1):
            train_loader = data[dset]['train']
            train_loss = train(model, epoch, train_loader)
            val_loader = data[dset]['val']
            val_loss = val(model, val_loader)
            losses[epoch - 1, 0] = train_loss
            losses[epoch - 1, 1] = val_loss

            # keep track of best weights -- this is equivalent
            # to a simple version of early-stopping
            '''
            is_best = best_acc < val_acc_meter.compute_gt_acc()
            if is_best:
                best_acc = val_acc_meter.compute_gt_acc()
                best_epoch = epoch

            # save weights to dict
            ### This really should just be taken from kwargs
            save_student_checkpoint(
                {
                    'state_dict': teacher.state_dict(),
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
        '''
        # plot loss over time
        plt.figure()
        plt.plot(range(args.epochs[dset_num]), losses[:, 0], '-', label='train')
        plt.plot(range(args.epochs[dset_num]), losses[:, 1], '-', label='val')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, model_id, '{}_loss_{}.png'.format(dset, dset_num)))

    '''
    print("Loading best model from disk ...")
    teacher = load_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'model_best.pth.tar'), use_cuda=args.cuda)
    train(best_epoch, False)
    val()
    '''

def output_samples_and_prototypes(speaker, train_loader, val_loader):
    train_samples = get_samples_and_prototypes(train_loader)
    train_samples.to_csv(os.path.join(args.out_dir, model_id, '{}_samples_train_{}.csv'.format(dset, dset_num)), index=False)
    val_samples = get_samples_and_prototypes(val_loader)
    val_samples.to_csv(os.path.join(args.out_dir, model_id, '{}_samples_val_{}.csv'.format(dset, dset_num)), index=False)
        

if __name__ == "__main__":
    args = parse_args()
    seed = set_seeds(args.seed)
    make_model_dir()
    kwargs = make_kwargs(args, seed)
    # TODO add back in normal single-model experiments
    if args.comm_dsets is not None:
        student_loaders, text_field = get_loaders(args.student_dsets, 'student')
        # Get student pretraining dataset loaders
        # Initialize student, train it on its pretraining dsets (initialize to use all vocab if not pretraining)
        student = Student(**kwargs)
        train_model(student)
        # If pretraining teacher, get teacher pretraining dataset loaders
        if args.pretrain_teacher:
            teacher_loaders = get_loaders(args.student_dsets, text_field=text_field)
            # Load in same datasets as for student pretraining
            # XXX First try using different field objects in the hopes that they use the same stoi
            # If not, use the same object, and figure out how not to share weights (it should be good enough to just have different torch embedding objs
            # XXX So maybe we can use the same field object after all :D
            # They'll both be initialized with glove, but that's okay (probably for the best)
        # Initialize teacher with same vocab as student (although making sure they don't share weights), and train it on the same pretraining dsets if desired
        #train_model(teacher, 
        #output_samples_and_prototypes(teacher, 
        # Run comm game
        # comm = CommunicationGame.train_comm_game(teacher, student, **kwargs)

        # TODO get samples (and prototypes :P) for teacher
