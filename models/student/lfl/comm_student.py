"""
File: multi_task_student.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 10, 2019
Description: Model of student who learns from language 
             -- playing both concept learning and reference games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_padded_sequence

from models.student.lfl.language.comm_bi_lstm import BiLSTM
from models.student.lfl.language.self_attention import SelfAttention
from models.student.lfl.mlp import MLP

import torchvision.models as models

class Student(nn.Module):
    """ Student takes langauge as input, develops a representation of the language and
        input stimulus, concatenates these representations together, and produces
        logits for the probability that the given stimulus belongs to the class described
        by the language.
    """
    def __init__(self, text_field, **kwargs):
        """
        @param **kwargs: parameters associated with initializing the language model
            and the stimulus model.
        @param text_field: the torchtext field defining the vocab to be used.
        """
        super(Student, self).__init__()

        self.languageModel = BiLSTM(
            h_dim=kwargs['h_dim_l_student'],
            o_dim=kwargs['o_dim_l_student'],
            d_prob=kwargs['d_prob_l_student'],      
            with_self_att=kwargs['with_self_att'],      
            d_dim=kwargs['d_dim_l'],      
            r_dim=kwargs['r_dim_l'],      
            num_layers=kwargs['num_layers_l_student'],      
            vocab_field=text_field,
        )
        
        if kwargs['stim_model_type'] == 'featureMLP':
            self.stimModel = MLP(
                i_dim=kwargs['i_dim_s_student'],
                h_dim=kwargs['h_dim_s_student'],
                o_dim=kwargs['o_dim_s_student'],
                d_prob=kwargs['d_prob_s_student'],
                batch_norm=kwargs['bn_student'],
                num_layers=kwargs['num_layers_s_student'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            self.stimModel = models.resnet18(pretrained=kwargs['pretrained'])
            self.stimModel.fc = nn.Linear(512, kwargs['o_dim_stim'])
        else:
            raise Exception('Invalid Stimulus Model')
            
        self.comparator = MLP(
            i_dim=kwargs['o_dim_l_student'] + kwargs['o_dim_s_student'],
            h_dim=kwargs['h_dim_student'],
            o_dim=1,
            d_prob=kwargs['d_prob_student'],
            batch_norm=kwargs['bn_student'],
            num_layers=kwargs['num_layers_student'],
        )

        self.cuda = kwargs['cuda']
        self.self_att = kwargs['self_att']
        # Store this for computing attention loss later
        self.r_dim = kwargs['r_dim_l']

    def forward(self, lang, stims, lang_lengths, onehot=False):
        """ lang: language (describing concept)
            stims: stimulus (from test set)
            lang_lengths: true lengths of text in lang
            use_concept: T/F use concept model
        """
        languageRep, alphas = self.languageModel(lang, lang_lengths, onehot=onehot)
        stimRep = self.stimModel(stims)
        joinedRep = torch.cat([languageRep, stimRep], dim=1)
        logits = self.comparator(joinedRep)
        return logits, alphas

    def compute_loss(self, batch, onehot=False):
        return self.compute_loss_cleaned(*(self.get_inputs_batch(batch, onehot=onehot)), onehot=onehot)

    def compute_loss_cleaned(self, stims, labels, lang, lang_lengths,
                             onehot=False):
        """ Compute reference game loss.
        """
        logits, alphas = self(lang, stims, lang_lengths, onehot=onehot)

        logits = logits.view(-1, 3)
        loss = F.cross_entropy(logits, labels)
        loss += self.compute_att_loss(alphas)
        return loss, logits

    def compute_att_loss(self, alphas):
        """ Compute loss term associated with self attention. 
        """
        if self.self_att:
            assert(alphas is not None), "Self Attention should have been applied"
            I = torch.eye(self.r_dim)
            if self.cuda:
                I = I.to('cuda')
            I = I.repeat(alphas.shape[0], 1, 1)
            alphas_t = torch.transpose(alphas, 1, 2).contiguous()
            extra_loss = torch.norm(torch.bmm(alphas, alphas_t) - I)
            return extra_loss
        else:
            return 0.0

    def compute_acc(self, logits):
        '''
        Computes accuracy of (for now, only) ref game predictions, stored in
        logits. Compares to self.labels, collected from last batch processed.
        logits: (batch, 3)
        self.labels: (batch)
        '''
        assert (logits.shape[0] == self.labels.shape[0]), \
                "Logits and labels not of the same shape!"
        preds = logits.argmax(1)
        correct = preds == self.labels
        n_correct = sum(correct).item()
        n_total = logits.shape[0]
        return n_correct, n_total

    ### HELPER METHODS ###
    def get_inputs_batch(self, batch, onehot=False):
        stims = self.construct_stim_reps(batch)
        return self.get_inputs(stims, batch.labels, batch.text[0], 
                          batch.text[1], onehot=onehot)

    def get_inputs(self, stims, labels, lang, lang_lengths, onehot=False):
        # TODO throw an error if they try to use elmo?
        max_msg_size = lang.shape[1]
        lang = torch.cat([lang]*3, dim=1) # repeat 3 times for 3 stims
        if onehot:
            lang = lang.view(-1, max_msg_size, lang.shape[2])
        else:
            lang = lang.view(-1, max_msg_size)

        lang_lengths = lang_lengths.unsqueeze(dim=1)
        lang_lengths = torch.cat([lang_lengths]*3, dim=1)
        lang_lengths = lang_lengths.view(-1, 1)
        lang_lengths = lang_lengths.squeeze()

        # Flatten stims (one per row)
        if len(stims.shape) > 2:
            n_feats = stims.shape[2]
            stims = stims.view(-1, n_feats)

        # Convert the one-hot labels into ints to use with cross-entropy
        labels = labels.argmax(1)
        if self.cuda:
            lang = lang.to('cuda')
            lang_lengths = lang_lengths.to('cuda')
            stims = stims.to('cuda')
            labels = labels.to('cuda')

        # Store labels of this batch to later compute accuracy
        self.labels = labels

        return stims, labels, lang, lang_lengths

    def construct_stim_reps(self, batch):
        """ 
        Cat stims from different columns into the same tensor
        returned with shape (batch, n_feats*3)
        Will be flattened later in the get_inputs call
        """
        target = batch.__dict__[str(0)]
        distr1 = batch.__dict__[str(1)]
        distr2 = batch.__dict__[str(2)]
        stims = torch.cat([target, distr1, distr2], dim=1).float()
        n_feats = target.shape[1]
        return stims.view(-1, n_feats) 
