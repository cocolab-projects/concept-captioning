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
        num_layers, vocab_field):
        """
        @param h_dim (int): hidden dimension of LSTM
        @param o_dim (int): dimension of the outputted language representation
        @param d_prob (float): probability that an output value should be dropped out
        @param with_self_att (bool): whether to utilize self attention or not
        @param d_dim (int): d_dim for self-attention
        @param r_dim (int): r_dim for self-attention
        @param num_layers (int): depth of LSTM
        @param vocab_field (torchtext.Field): Field with vocab object
        """
        super(BiLSTM, self).__init__()
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.d_dim = d_dim
        self.r_dim = r_dim
        self.d_prob = d_prob
        self.num_layers = num_layers
        self.with_self_att = with_self_att
        self.vocab_field = vocab_field

        # Layer Initialization 
        self.load_embeddings(vocab_field)
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


    def load_embeddings(self, vocab_field):
        """
        Loads embeddings from a torchtext vocab field into an embedding module.
        """
        # Embedding initialization
        self.embed_dim = self.vocab_field.vocab.vectors.shape[1]
        self.embed = nn.Embedding(
            len(self.vocab_field.vocab),
            self.embed_dim,
            padding_idx=self.vocab_field.vocab.stoi[self.vocab_field.pad_token]
        )
        self.embed.weight.data.copy_(self.vocab_field.vocab.vectors)

    def forward(self, x, lang_lengths, onehot=False):
        """
        @param x (torch.Tensor): input text tensor (b, l) where b is batch size,
            l is largest input sequence length (after padding)
        @param batch_lengths (torch.Tensor): true lengths of inputs in x

        @returns language_rep (torch.Tensor): representation of inputted x (b, o_dim) where
            b is batch size and o_dim is output dimension
        """

        if onehot:
            # (b, max_seq_len, n_vocab) X (n_vocab, embed_dim) -> (b, max_seq_len, embed_dim)
            embeddings = x @ self.embed.weight
        else:
            embeddings = self.embed(x)
        sorted_lengths, sorted_idx = torch.sort(lang_lengths, descending=True)
        embeddings = embeddings[sorted_idx]

        X = pack_padded_sequence(embeddings, sorted_lengths, batch_first=True) # (b, l, w_e)
        
        hiddens, (state, _) = self.lstm(X)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True) # (b, l, h_d * 2)

        # TODO check that this should be sorted_lengths, and that we even need it to be a list at all
        lang_lengths = sorted_lengths.view(-1).tolist()
        if self.with_self_att:
            language_att, alphas = self.attention_model(hiddens, lang_lengths)
            language_rep = self.output_projection(language_att)
        else:
            state = torch.cat([state[0], state[1]], dim=1)
            language_rep = torch.tanh(self.output_projection(self.dropout(state)))
            alphas = None

        return language_rep, alphas
