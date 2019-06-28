from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import dill
import numpy as np
import random
import os
import shutil


import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils.constants import Constants
from utils.dataloaders.vectorized.reference_load_dataset import construct_y as construct_y_reference
from utils.dataloaders.vectorized.concept_load_dataset import construct_y as construct_y_concept
from models.student.lfl.single_task_student import SingleTaskStudent
from models.student.lfl.multi_task_student import MultiTaskStudent
from collections import defaultdict, Counter, OrderedDict


# --------------
# STUDENT MODELS
# --------------

def save_student_checkpoint(state, is_best, folder='./',
    filename='checkpoint.pth.tar', best_filepath='model_best.pth.tar'):
    """ Save checkpoint of student model.
    """
    dir = os.path.join(folder, 'model_weights')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    torch.save(state, os.path.join(dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir, filename),
                        os.path.join(dir, best_filepath))


def load_single_task_student_checkpoint(file_path, use_cuda=False, refGame=False):
    """ Load student checkpoint.
    """
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    with open(checkpoint['vocab_file'], 'rb') as input:
        vocab = torch.load(input, pickle_module=dill) if use_cuda else \
            torch.load(file_path, map_location=lambda storage, location: storage, pickle_module=dill)
    if refGame:
        checkpoint['kwargs']['reference_vocab_field'] = vocab
    else:
        checkpoint['kwargs']['concept_vocab_field'] = vocab

    model = SingleTaskStudent(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['state_dict'])

    # this way we always know the settings used to train the model.
    model.kwargs = checkpoint['kwargs']

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = model.to(device)
    return model


def load_multi_task_student_checkpoint(file_path, use_cuda=False):
    """ Load student checkpoint.
    """
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    with open(checkpoint['ref_vocab_file'], 'rb') as input:
        ref_vocab = torch.load(input, pickle_module=dill) if use_cuda else \
            torch.load(input, map_location=lambda storage, location: storage, pickle_module=dill)
    checkpoint['kwargs']['reference_vocab_field'] = ref_vocab

    with open(checkpoint['concept_vocab_file'], 'rb') as input:
        concept_vocab = torch.load(input, pickle_module=dill) if use_cuda else \
            torch.load(input, map_location=lambda storage, location: storage, pickle_module=dill)    
    checkpoint['kwargs']['concept_vocab_field'] = concept_vocab

    model = MultiTaskStudent(**checkpoint['kwargs'])
    checkpoint['state_dict']['languageModel.concept_embed.weight'] = model.languageModel.concept_embed.weight
    checkpoint['state_dict']['languageModel.ref_embed.weight'] =  model.languageModel.ref_embed.weight
    model.load_state_dict(checkpoint['state_dict'])

    # this way we always know the settings used to train the model.
    model.kwargs = checkpoint['kwargs']

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = model.to(device)
    return model


# --------------
# GENERAL UTILS
# --------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """ Computes accuracy against 3 "truth" values:
            1. Ground Truth
            2. Teacher Response
            3. Student Response
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.acc_counts = {
            Constants.TRAIN_OBJECTIVES['ground_truth']: {
                'num_correct': 0,
                'num_incorrect': 0, 
            },
            Constants.TRAIN_OBJECTIVES['teacher']: {
                'num_correct': 0,
                'num_incorrect': 0, 
            },
            Constants.TRAIN_OBJECTIVES['student']: {
                'num_correct': 0,
                'num_incorrect': 0, 
            },
        }
    
    def update(self, y_hat, batch, refGame=False, cuda=False, vision=False):
        predictions = torch.argmax(y_hat, dim=1)
        for obj in self.acc_counts.keys():
            if vision:
                if refGame:
                    raise Exception ("Not implemented yet.")
                else:
                    acc_count_y = batch[obj] # Labels
            else:
                if refGame:
                    acc_count_y = construct_y_reference(batch)
                    if cuda:
                        acc_count_y = acc_count_y.to('cuda')                
                else:
                    acc_count_y = construct_y_concept(batch, obj)
            num_incorrect = (acc_count_y - predictions).nonzero()
            if not isinstance(num_incorrect, int):
                num_incorrect = num_incorrect.squeeze(dim=0).shape[0]
            num_correct = predictions.shape[0] - num_incorrect
            self.acc_counts[obj]['num_correct'] += num_correct
            self.acc_counts[obj]['num_incorrect'] += num_incorrect
    

    def compute_gt_acc(self):
        return self.acc_counts['ground_truth']['num_correct'] * 1.0 / (self.acc_counts['ground_truth']['num_correct'] + self.acc_counts['ground_truth']['num_incorrect'])


    def compute_t_acc(self):
        return self.acc_counts['teacher']['num_correct'] * 1.0 / (self.acc_counts['teacher']['num_correct'] + self.acc_counts['teacher']['num_incorrect'])


    def compute_s_acc(self):
        return self.acc_counts['student']['num_correct'] * 1.0 / (self.acc_counts['student']['num_correct'] + self.acc_counts['student']['num_incorrect'])


    def print(self, ground_truth_only=False):
        print('\t Ground Truth Accuracy: {}'.format(self.compute_gt_acc()))
        if not ground_truth_only:
            print('\t Teacher Accuracy: {}'.format(self.compute_t_acc()))
            print('\t Student Accuracy: {}'.format(self.compute_s_acc()))


def set_seeds(seed=12132):
    """ Set random seeds to ensure result reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reverse(batch, vocabField, refGame=False):
    """ Reverse messages represented as integer indices into
        plain text.
    """
    if refGame:
        msgs = batch.message[0].tolist()
    else:
        msgs = batch.text[0].tolist()
    y = [[vocabField.vocab.itos[ind] for ind in ex] for ex in msgs]  # denumericalize
    return y
