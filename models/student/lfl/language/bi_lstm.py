"""
File: bi_lstm.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 21, 2019
Description: Bidirectional LSTM for language understanding.
"""

from models.student.lfl.language.self_attention import SelfAttention

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    """ Bidirectional LSTM with optional self attention. 
    """
    def __init__(self, h_dim, o_dim, d_prob, with_self_att, d_dim, r_dim,
        num_layers, embeddings, concept_vocab_field=None, ref_vocab_field=None, task=''):
        """
        @param h_dim (int): hidden dimension of LSTM
        @param o_dim (int): dimension of the outputted language representation
        @param d_prob (float): probability that an output value should be dropped out
        @param with_self_att (bool): whether to utilize self attention or not
        @param d_dim (int): d_dim for self-attention
        @param r_dim (int): r_dim for self-attention
        @param num_layers (int): depth of LSTM
        @param embeddings (string): embedding type ["glove", "elmo"]
        @param concept_vocab_field (torchtext.Field): Field with vocab object
        @param ref_vocab_field (torchtext.Field): Field with vocab object

        Note: If only concept_vocab_field or ref_vocab_field is provided, we default
                to only having one vocabulary.
        """
        super(BiLSTM, self).__init__()
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.d_dim = d_dim
        self.r_dim = r_dim
        self.d_prob = d_prob
        self.num_layers = num_layers
        self.with_self_att = with_self_att
        self.embeddings = embeddings
        self.task = task

        # Layer Initialization 
        self.replace_embeddings(concept_vocab_field, ref_vocab_field)
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.h_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=self.d_prob)
        if self.with_self_att:
            self.output_projection = nn.Linear(
                self.h_dim * 2 * self.r_dim, # must account for attention hops
                self.o_dim,
                bias=False
            )
            self.attention_model = SelfAttention(
                self.h_dim*2,
                self.d_dim,
                self.r_dim,
                self.dropout
            )
        else:
            self.attention_model = None
            self.output_projection = nn.Linear(
                self.h_dim * 2,
                self.o_dim,
                bias=False
            )


    def replace_embeddings(self, concept_vocab_field=None, ref_vocab_field=None):
        """ Assuming that the word vectors are frozen GLoVe vectors
            one may have slight variations in the necessary embeddings
            -- between train/val/test splits. Here, we allow for a 
            different subset of word vectors to be loaded up from GLoVe
            depending on the provided vocab_field entry.

            If this is a single task model
        """
        if self.task == 'multi':
            if ref_vocab_field is None:
                raise Exception('Must pass in a Reference Game Vocab')
            if concept_vocab_field is None:
                raise Exception('Must pass in a Concept Game Vocab')
            self.concept_vocab_field = concept_vocab_field
            self.ref_vocab_field = ref_vocab_field
        elif self.task == 'ref':
             if ref_vocab_field is None:
                raise Exception('Must pass in a Reference Game Vocab')    
             self.vocab_field = ref_vocab_field       
        elif self.task == 'concept':
            if concept_vocab_field is None:
                raise Exception('Must pass in a Concept Game Vocab')
            self.vocab_field = concept_vocab_field
        else:
            raise Exception('Invalid task')

        # Embedding initialization
        if self.task == 'ref' or self.task == 'concept':
            self.embed_dim = self.vocab_field.vocab.vectors.shape[1]
            self.embed = nn.Embedding(
                len(self.vocab_field.vocab),
                self.embed_dim,
                padding_idx=self.vocab_field.vocab.stoi[self.vocab_field.pad_token]
            )
            self.embed.weight.data.copy_(self.vocab_field.vocab.vectors)
        elif self.task == 'multi':
            self.embed_dim = self.concept_vocab_field.vocab.vectors.shape[1]
            self.concept_embed = nn.Embedding(
                len(self.concept_vocab_field.vocab),
                self.embed_dim,
                padding_idx=self.concept_vocab_field.vocab.stoi[self.concept_vocab_field.pad_token]
            )
            self.concept_embed.weight.data.copy_(self.concept_vocab_field.vocab.vectors)
            self.ref_embed = nn.Embedding(
                len(self.ref_vocab_field.vocab),
                self.embed_dim,
                padding_idx=self.ref_vocab_field.vocab.stoi[self.ref_vocab_field.pad_token]
            )
            self.ref_embed.weight.data.copy_(self.ref_vocab_field.vocab.vectors)       
    

    def forward(self, x, batch_lengths=None, use_concept_vocab=None):
        """
        @param x (torch.Tensor): input text tensor (b, l) where b is batch size,
            l is largest input sequence length (after padding)
        @param batch_lengths (torch.Tensor): true lengths of inputs in x
        @param use_concept_vocab (bool): If BiLSTM is being utilized as part of a multitask
            model, we then have to state which vocabulary to utilize.

        @returns language_rep (torch.Tensor): representation of inputted x (b, o_dim) where
            b is batch size and o_dim is output dimension
        """

        batch_lengths = batch_lengths.view(-1).tolist()
        if self.task in ['ref', 'concept']:
            X = pack_padded_sequence(self.embed(x), batch_lengths, batch_first=True) # (b, l, w_e)
        elif self.task == 'multi':
            if use_concept_vocab is None:
                raise Exception('Multitask BiLSTM must specify whether to us concept or reference vocabulary.')
            elif use_concept_vocab:
                X = pack_padded_sequence(self.concept_embed(x), batch_lengths, batch_first=True) # (b, l, w_e)
            else:
                X = pack_padded_sequence(self.ref_embed(x), batch_lengths, batch_first=True) # (b, l, w_e)
        else:
            raise Exception('Undefined task') 

        hiddens, (state, _) = self.lstm(X)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True) # (b, l, h_d * 2)

        if self.with_self_att:
            language_att, alphas = self.attention_model(hiddens, batch_lengths)
            language_rep = self.output_projection(language_att)
        else:
            state = torch.cat([state[0], state[1]], dim=1)
            language_rep = torch.tanh(self.output_projection(self.dropout(state)))
            alphas = None
        
        return language_rep, alphas
