"""
File: mlp.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 22, 2019
Description: Multilayer Perceptron, used for stimulus feature representations and predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class MLP(nn.Module):
    """ MLP that produces a representation of stimuli, which initially
        have featurized representations.
    """
    def __init__(self, i_dim, h_dim, o_dim, d_prob, batch_norm=False, num_layers=2):
        """
        @param i_dim (int): input dimension
        @param h_dim (int): hidden dimension
        @param o_dim (int): output dimension
        @param d_prob (float): probability that an output value should be dropped out
        @param batch_norm (bool): whether or not to use batch normalization on input
        @param num_layers(int): number of layers in network
        """
        super(MLP, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.d_prob = d_prob

        if num_layers < 1:
            raise Exception('Must utilize at least one layer')
        elif num_layers == 1:
            # Overview:
            #     Leaky Relu
            #     Dropout
            #     Linear layer: input dim -> output dim
            self.network_components = nn.ModuleList([nn.LeakyReLU()])
            if batch_norm:
                self.network_components.append(torch.nn.BatchNorm1d(self.h_dim))
            self.network_components.append(nn.Dropout(p=self.d_prob))
            self.network_components.append(nn.Linear(self.i_dim, self.o_dim))
        else:
            # Overview:
            #   Linear Layer: input dim -> hidden dim
            #   Leaky Relu
            #   Dropout
            #   num_layers - 2 of:
            #       Linear Layer: hidden dim -> hidden dim
            #       Leaky Relu
            #       Dropout
            #   Linear Layer: hidden dim -> output dim 

            self.network_components = nn.ModuleList([nn.Linear(self.i_dim, self.h_dim)])
            self.network_components.append(nn.LeakyReLU())
            if batch_norm:
                self.network_components.append(torch.nn.BatchNorm1d(self.h_dim))
            self.network_components.append(nn.Dropout(p=self.d_prob))
            for h_i in range(num_layers-2):
                self.network_components.append(nn.Linear(self.h_dim, self.h_dim))
                self.network_components.append(nn.LeakyReLU())
                if batch_norm:
                    self.network_components.append(torch.nn.BatchNorm1d(self.h_dim))
                self.network_components.append(nn.Dropout(p=self.d_prob))
            self.network_components.append(nn.Linear(self.h_dim, self.o_dim))

    def forward(self, x):
        """
        @param x (torch.Tensor): input stimulus feature representation (b, i_dim)
            where b is batch size, i is the input dimension (dim of stimulus representation)

        @returns logits (torch.Tensor): representation of inputted x (b, o_dim) where
            b is batch size and o_dim is output dimension
        """
        logits = x
        for i, l in enumerate(self.network_components):
            logits = l(logits)
        return logits
