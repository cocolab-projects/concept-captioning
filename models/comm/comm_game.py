"""
File: comm_game.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: August 5, 2019
Description: Communication game model to train a teacher and student to
             describe concepts or objects to each other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_padded_sequence
import torchvision.models as models
import numpy as np
from contextlib import nullcontext

from models.student.lfl.mlp import MLP
from models.teacher.language.rnn_decoder import RNNDecoder

class CommGame(nn.Module):
    """
    Trains a teacher to describe concepts to or play reference games with
    a student
    """ 
    def __init__(self, teacher, student, **kwargs):
        """
        @param **kwargs: parameters associated with initializing the language model
            and the stimulus model.
        """
        super(CommGame, self).__init__()

        self.teacher = teacher
        self.student = student
        
        # self.cuda = kwargs['cuda']
        # self.embeddings = kwargs['embeddings']
        # self.start_index = kwargs['start_index']
        # self.end_index = kwargs['end_index']
        # self.pad_index = kwargs['pad_index']
        # self.text_field = kwargs['vocab_field']
            
    # def forward(self, batch):
    #     """
    #     Inputs: batch, a batch of concept or reference game inputs
    #     Outputs: 
    #     """

    def compute_loss(self, batch, n_ref_games=0, fix_student=False):
        """
        Compute loss.
        """
        # TODO change this to Jesse's param freezing and/or check if that's necessary
        if fix_student:
            self.student.eval()
        else:
            self.student.train()
        # Get gumbel-softmax one-hot samples from teacher
        # Shape: (batch, max seq length, vocab size)
        stims, labels, _, _ = self.teacher.get_inputs(batch)
        lang, lang_lengths = self.teacher.sample(stims, labels, train=True)
        # If running on a concept batch (i.e. n_ref_games != 0), 
        # then create a batch of ref games (some number for each concept)
        # ("create" just a single ref game if running on ref data already)
        # For each game, input the teacher sample and ref game to the student,
        # and get its classification loss (run this part with torch.no_grad())
        cm = torch.no_grad() if fix_student else nullcontext()
        batch.text = (lang, lang_lengths)
        with cm:
            # loss = self.student.compute_loss_cleaned(stims, labels, lang, lang_lengths, onehot=True)
            loss = self.student.compute_loss(batch, onehot=True)
        # Sum across all games to get the total loss and return it
        return loss
