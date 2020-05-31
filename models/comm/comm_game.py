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
        self.fix_student = kwargs['fix_student']

        # Labels from last compute_loss(), i.e. last batch of ref games
        # Stored in order to later compute accuracy
        # Shape: (batch, n_ref_games)
        self.labels = None
        
    def compute_loss(self, batch, n_ref_games=0):
        """
        Compute student loss for a batch of either concepts or ref games.
        If n_ref_games == 0, it's assumed that the batch is of ref games;
        if n_ref_games >= 0, it may still be ref games, but "new" ref games
        will be sampled from the original, i.e. distractors may be reused.
        """
        # TODO change this to Jesse's param freezing and/or check if that's necessary
        if self.fix_student:
            for param in self.student.parameters():
                param.requires_grad = False
        # Get gumbel-softmax one-hot samples from teacher
        stims, labels, _, _ = self.teacher.get_inputs(batch)
        lang, lang_lengths = self.teacher.sample(stims, labels, train=True)
        # lang shape: (batch, max_lang_len, n_vocab)
        # lang_lengths shape: (batch)
        # If running on a concept batch (i.e. n_ref_games != 0), 
        # then create a batch of ref games (some number for each concept)
        # ("create" just a single ref game if running on ref data already)
        # For each game, input the teacher sample and ref game to the student,
        # and get its classification loss (run this part with torch.no_grad())
        # XXX apparently can't use torch.no_grad() for this
        if n_ref_games > 0:
            return self.compute_concept_loss(stims, labels, lang, 
                                             lang_lengths, n_ref_games)
        else: 
            inputs_clean = self.student.get_inputs(stims, batch.labels,
                                                   lang, lang_lengths,
                                                   onehot=True)
            self.labels = inputs_clean[1]
            return self.student.compute_loss_cleaned(*inputs_clean,
                                                    onehot=True)

    def compute_concept_loss(self, stims, labels, lang, lang_lengths, 
                             n_ref_games):
        '''
        Computes student loss on a set of reference games generated from
        a concept.
        '''
        bsize, n_feats = stims.shape[0], stims.shape[2]
        # Take multinomial sample from nonzero elements of labels,
        # i.e. the positive examples of the concept (or ref game targets)
        target_indices = labels.multinomial(n_ref_games, replacement=True)
        # target_indices shape: (batch, n_ref_games)
        # Index into each tensor of stimuli using the indices above
        target_indices_expanded = target_indices.unsqueeze(2).expand(-1, -1, n_feats)
        targets = stims.gather(1, target_indices_expanded)
        targets = targets.unsqueeze(2)
        # targets shape: (batch, n_ref_games, 1, n_feats)
        # TODO with replacement=True this won't exactly reproduce ref games
        # Could be a useful normalization technique, but should talk about it
        distr_indices = (1-labels).multinomial(n_ref_games*2, replacement=True)
        # shape: (batch, n_ref_games*2)
        distr_indices = distr_indices.unsqueeze(2).expand(-1, -1, n_feats)
        distr = stims.gather(1, distr_indices)
        # shape: (batch, n_ref_games*2, n_feats)
        distr = distr.view(-1, n_ref_games, 2, n_feats)
        ref_games = torch.cat((targets, distr), dim=2)
        # shape: (batch, n_ref_games, 3, n_feats)
        breakpoint()

        # Pivot games into all one dimension for feeding into student
        ref_games_flat = self.ref_games.view(-1, 3, n_feats)
        # Target indices are just those generated randomly above
        # (still have to pivot into one dimension)
        self.labels = target_indices.long().view(-1)
        # Repeat lang and lang_lengths for each reference game,
        # and each stimulus within each reference game
        lang = torch.cat([lang]*n_ref_games*3, dim=1)
        lang = lang.view(bsize*n_ref_games*3, lang.shape[1], -1)
        # shape: (batch*n_ref_games*3, max_lang_len, n_vocab)
        lang_lengths = torch.cat([lang_lengths.unsqueeze(1)]*n_ref_games*3, dim=1)
        lang_lengths = lang_lengths.view(-1)
        # shape: (batch*n_ref_games*3)

        loss, logits = self.student.compute_loss_cleaned(ref_games_flat,
                                                         self.labels,
                                                         lang,
                                                         lang_lengths,
                                                         onehot=True)
        # Sum across all games to get the total loss and return it
        # logits: (batch, n_ref_games, 3)
        # shapely_logits = logits.view(bsize, n_ref_games, 3)
        # TODO decide what shape of logits to return
        # (More info, or more consistent with the ref teacher shape?)
        # Current shape: (bsize*n_ref_games, 3)
        return loss, logits

    def compute_acc(self, logits):
        '''
        Computes the accuracy of the given logits for solving the last
        set of reference games, stored in self.labels.
        self.labels: (batch*n_ref_games). Stored from last compute_loss().
        logits: (batch*n_ref_games, 3).
        NOTE: as suggested by the sizes, assumes that logits are flattened.
        '''
        assert (self.labels.shape[0] == logits.shape[0]), \
                       "Logit and label shapes don't match!"

        preds = logits.argmax(1) # (batch, n_ref_games)
        correct = preds == self.labels # (batch*n_ref_games)
        n_correct = sum(correct).item()
        n_total = preds.shape[0]
        return n_correct, n_total
