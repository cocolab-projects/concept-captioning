"""
File: teacher.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 9, 2019
Description: Model of a teacher who describes concepts with language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_padded_sequence
import torchvision.models as models

from models.student.lfl.language.bi_lstm import BiLSTM
from models.student.lfl.language.self_attention import SelfAttention
from models.student.lfl.mlp import MLP

class Teacher(nn.Module):
    """ Teacher takes a concept (a set of images/feature vectors labeled as positive or negative)
    and outputs a natural language description of that concept    """
    def __init__(self, **kwargs):
        """
        @param **kwargs: parameters associated with initializing the language model
            and the stimulus model.
        """
        super(Teacher, self).__init__()
        # if kwargs['language_model_type'] == 'bilstm':
        #    self.languageModel = BiLSTM(
        #         h_dim=kwargs['h_dim_l'],
        #         o_dim=kwargs['o_dim_l'],
        #         d_prob=kwargs['d_prob_l'],      
        #         with_self_att=kwargs['with_self_att'],      
        #         d_dim=kwargs['d_dim_l'],      
        #         r_dim=kwargs['r_dim_l'],      
        #         num_layers=kwargs['num_layers_l'],      
        #         embeddings=kwargs['embeddings'],
        #         concept_vocab_field=kwargs['concept_vocab_field'],
        #         ref_vocab_field=kwargs['reference_vocab_field'],
        #         task=kwargs['task']
        #     )
        # else:
        #     raise Exception('Invalid Language Model')
        
        ### Either way, stimModel outputs an embedded representation of the its inputs
        if kwargs['stim_model_type'] == 'featureMLP':
            ### Indicates that we're using vectorized features
            ### Note that MLP is defined manually
            self.stimModel = MLP(
                ### Input, hidden and output dimensions
                i_dim=kwargs['i_dim_s'],
                h_dim=kwargs['h_dim_s'],
                o_dim=kwargs['o_dim_s'],
                ### Dropout probability for dropout layers
                d_prob=kwargs['d_prob_s'],
                ### Whether or not to use batch normalization, where node activations are normalized by the batch mean and SD
                ### This can mitigate negative effects on training speed of high variability of internal activations
                batch_norm=kwargs['batch_norm'],
                num_layers=kwargs['num_layers_s'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            ### Indicates that we're using images; resnet (residual neural network) is a CNN
            self.stimModel = models.resnet18(pretrained=kwargs['pretrained'])
            self.stimModel.fc = nn.Linear(512, kwargs['o_dim_s'])

        else:
            raise Exception('Invalid Stimulus Model')
            
        ### MLP to go from language rep + stimulus rep to probability that 
        ### the concept description applies to the stimulus
        # self.mlp = MLP(
        #     i_dim=kwargs['o_dim_l'] + kwargs['o_dim_s'],
        #     h_dim=kwargs['hidden_dim_student'],
        #     o_dim=kwargs['output_dim'],
        #     d_prob=kwargs['d_prob_student'],
        #     batch_norm=kwargs['batch_norm'],
        #     num_layers=kwargs['num_layers_student'],

    ### forward: defines how the module transforms input to output
    ### technically just a function on any arbitrary input, can give any arbitrary output?
    ### called using the name of the object, e.g. teacher(input)
    ### (i.e. they override the __call__ function)
    def forward(self, stims, labels, language, lang_lengths):
        """ stims: (batch_size, num_examples, num_features)
            labels: (batch_size, num_examples)
            language: (batch_size, max_lang_length)
            lang_lengths: (batch_size)
        """
        # Create representations of stimuli
        batch_size, num_examples, num_features = stims.shape
        stims_flat = stims.view(batch_size*num_examples, num_features)
        stim_reps_flat = self.stimModel(stims_flat)
        # stim_reps_flat: (batch_size*num_examples, rep_length)
        stim_reps = stim_reps_flat.unsqueeze(dim = 1)
        stim_reps = stim_reps.view(batch_size, num_examples, -1)
        # stim_reps: (batch_size, num_examples, rep_length)
        stim_reps = stim_reps.permute(0, 2, 1)
        # stim_reps: (batch_size, rep_length, num_examples)
        labels_mat = labels.unsqueeze(dim = 2)
        # labels: (batch_size, num_examples, 1)
        ### bmm: stands for batch matrix multiplication
        pos_prototypes = torch.bmm(stim_reps, labels_mat) # (batch_size, rep_length, 1)
        pos_prototypes = pos_prototypes.squeeze(dim = 2)
        neg_prototypes = torch.bmm(stim_reps, 1-labels_mat)
        neg_prototypes = neg_prototypes.squeeze(dim = 2)
        breakpoint()
        ### n_pos: number of positive examples of a concept (i.e. number of 1 labels)
        n_pos = labels.sum(dim = 1) # (batch_size)
        pos_prototypes = pos_prototypes / n_pos.unsqueeze(1).expand_as(pos_prototypes) # same dimension
        n_neg = (1 - labels).sum(dim = 1)
        neg_prototypes = neg_prototypes / n_neg.unsqueeze(1).expand_as(pos_prototypes)
    
        
        # Form representation of concept from stimuli
        # Eventually to be done with attention, but for now just uses averages
        # TODO average 
        
        logits = self.mlp(joinedRep)
        return logits, alphas


        # joinedRep = torch.cat([languageRep, stimRep], dim=1) 
        ### above, dim is the concatenating dimension;
        ### in this case, dim 0 is likely batches, so dim 1 is the reps themselves
        # languageRep, alphas = self.languageModel(x1, x1_lengths)
