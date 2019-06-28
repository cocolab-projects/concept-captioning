"""
File: single_task_student.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 21, 2019
Description: Model of student who learns from language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_padded_sequence
import torchvision.models as models

from models.student.lfl.language.bi_lstm import BiLSTM
from models.student.lfl.language.self_attention import SelfAttention
from models.student.lfl.mlp import MLP

class SingleTaskStudent(nn.Module):
    """ Student takes langauge as input, develops a representation of the language and
        input stimulus, concatenates these representations together, and produces
        logits for the probability that the given stimulus belongs to the class described
        by the language.
    """
    def __init__(self, **kwargs):
        """
        @param **kwargs: parameters associated with initializing the language model
            and the stimulus model.
        """
        super(SingleTaskStudent, self).__init__()
        if kwargs['language_model_type'] == 'bilstm':
            self.languageModel = BiLSTM(
                h_dim=kwargs['h_dim_l'],
                o_dim=kwargs['o_dim_l'],
                d_prob=kwargs['d_prob_l'],      
                with_self_att=kwargs['with_self_att'],      
                d_dim=kwargs['d_dim_l'],      
                r_dim=kwargs['r_dim_l'],      
                num_layers=kwargs['num_layers_l'],      
                embeddings=kwargs['embeddings'],
                concept_vocab_field=kwargs['concept_vocab_field'],
                ref_vocab_field=kwargs['reference_vocab_field'],
                task=kwargs['task']
            )
        else:
            raise Exception('Invalid Language Model')
        
        if kwargs['stim_model_type'] == 'featureMLP':
            self.stimModel = MLP(
                i_dim=kwargs['i_dim_s'],
                h_dim=kwargs['h_dim_s'],
                o_dim=kwargs['o_dim_s'],
                d_prob=kwargs['d_prob_s'],
                batch_norm=kwargs['batch_norm'],
                num_layers=kwargs['num_layers_s'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            self.stimModel = models.resnet18(pretrained=kwargs['pretrained'])
            self.stimModel.fc = nn.Linear(512, kwargs['o_dim_s'])

        else:
            raise Exception('Invalid Stimulus Model')
            
        self.mlp = MLP(
            i_dim=kwargs['o_dim_l'] + kwargs['o_dim_s'],
            h_dim=kwargs['hidden_dim_student'],
            o_dim=kwargs['output_dim'],
            d_prob=kwargs['d_prob_student'],
            batch_norm=kwargs['batch_norm'],
            num_layers=kwargs['num_layers_student'],
        )

    def forward(self, x1, x2, x1_lengths):
        """ x1: language (describing concept)
            x2: stimulus (from test set)
            x1_lengths: true lengths of text in x1
        """
        languageRep, alphas = self.languageModel(x1, x1_lengths)
        stimRep = self.stimModel(x2)
        joinedRep = torch.cat([languageRep, stimRep], dim=1)
        logits = self.mlp(joinedRep)
        return logits, alphas
