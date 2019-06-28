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

from models.student.lfl.language.bi_lstm import BiLSTM
from models.student.lfl.language.self_attention import SelfAttention
from models.student.lfl.mlp import MLP

import torchvision.models as models

class MultiTaskStudent(nn.Module):
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
        super(MultiTaskStudent, self).__init__()
        if kwargs['language_model_type'] == 'bilstm':
            self.languageModel = BiLSTM(
                h_dim=kwargs['h_dim_lang'],
                o_dim=kwargs['o_dim_lang'],
                d_prob=kwargs['dropout_lang'],      
                with_self_att=kwargs['self_att'],      
                d_dim=kwargs['d_dim_lang'],      
                r_dim=kwargs['r_dim_lang'],      
                num_layers=kwargs['num_layers_lang'],      
                embeddings=kwargs['embeddings'],
                concept_vocab_field=kwargs['concept_vocab_field'],
                ref_vocab_field=kwargs['reference_vocab_field'],
                task=kwargs['task']
            )
        else:
            raise Exception('Invalid Language Model')
        
        if kwargs['stim_model_type'] == 'featureMLP':
            self.stimModel = MLP(
                i_dim=kwargs['i_dim_stim'],
                h_dim=kwargs['h_dim_stim'],
                o_dim=kwargs['o_dim_stim'],
                d_prob=kwargs['dropout_stim'],
                batch_norm=kwargs['bn'],
                num_layers=kwargs['num_layers_stim'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            self.stimModel = models.resnet18(pretrained=kwargs['pretrained'])
            self.stimModel.fc = nn.Linear(512, kwargs['o_dim_stim'])
        else:
            raise Exception('Invalid Stimulus Model')
            
        self.shared_rep_model = MLP(
            i_dim=kwargs['o_dim_lang'] + kwargs['o_dim_stim'],
            h_dim=kwargs['h_dim_shared_student'],
            o_dim=kwargs['h_dim_shared_student'],
            d_prob=kwargs['dropout_shared_student'],
            batch_norm=kwargs['bn'],
            num_layers=kwargs['num_layers_shared_student'],
        )

        if kwargs['task'] == 'multi' or kwargs['task'] == 'ref':
            self.ref_model = MLP(
                i_dim=kwargs['h_dim_shared_student'],
                h_dim=kwargs['h_dim_ref_student'],
                o_dim=1,
                d_prob=kwargs['dropout_ref_student'],
                batch_norm=kwargs['bn'],
                num_layers=kwargs['num_layers_ref_student'],            
            )

        if kwargs['task'] == 'multi' or kwargs['task'] == 'concept':
            self.concept_model = MLP(
                i_dim=kwargs['h_dim_shared_student'],
                h_dim=kwargs['h_dim_concept_student'],
                o_dim=2,
                d_prob=kwargs['dropout_concept_student'],
                batch_norm=kwargs['bn'],
                num_layers=kwargs['num_layers_concept_student'],            
            )


    def forward(self, x1, x2, x1_lengths, use_concept=False):
        """ x1: language (describing concept)
            x2: stimulus (from test set)
            x1_lengths: true lengths of text in x1
            use_concept: T/F use concept model
        """
        languageRep, alphas = self.languageModel(x1, x1_lengths, use_concept_vocab=use_concept)
        stimRep = self.stimModel(x2)
        joinedRep = torch.cat([languageRep, stimRep], dim=1)
        sharedRep = self.shared_rep_model(joinedRep)
        if use_concept:
            logits = self.concept_model(sharedRep)
        else:
            logits = self.ref_model(sharedRep)
        return logits, alphas
