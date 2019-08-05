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
import numpy as np

from models.student.lfl.mlp import MLP
from models.teacher.language.rnn_decoder import RNNDecoder

class Teacher(nn.Module):
    """ Teacher takes a concept (a set of images/feature vectors labeled as positive or negative)
    and outputs a natural language description of that concept    """
    def __init__(self, **kwargs):
        """
        @param **kwargs: parameters associated with initializing the language model
            and the stimulus model.
        """
        super(Teacher, self).__init__()
        
        self.decoder = RNNDecoder(**kwargs)
        ### Either way, stimModel outputs an embedded representation of the its inputs
        if kwargs['stim_model_type'] == 'featureMLP':
            ### Indicates that we're using vectorized features
            ### Note that MLP is defined manually
            self.stimModel = MLP(
                ### Input, hidden and output dimensions
                i_dim=kwargs['i_dim_s_teacher'],
                h_dim=kwargs['h_dim_s_teacher'],
                o_dim=kwargs['o_dim_s_teacher'],
                ### Dropout probability for dropout layers
                d_prob=kwargs['d_prob_s_teacher'],
                ### Whether or not to use batch normalization, where node activations are normalized by the batch mean and SD
                ### This can mitigate negative effects on training speed of high variability of internal activations
                batch_norm=kwargs['bn_teacher'],
                num_layers=kwargs['num_layers_s_teacher'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            ### Indicates that we're using images; resnet (residual neural network) is a CNN
            self.stimModel = models.resnet18(pretrained=kwargs['pretrained'])
            self.stimModel.fc = nn.Linear(512, kwargs['o_dim_s'])

        else:
            raise Exception('Invalid Stimulus Model')

        self.cuda = kwargs['cuda']
        self.embeddings = kwargs['embeddings']
        self.start_index = kwargs['start_index']
        self.end_index = kwargs['end_index']
        self.pad_index = kwargs['pad_index']
        self.text_field = kwargs['vocab_field']
            
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
        pos_prototypes, neg_prototypes = self.get_prototypes(stims, labels)
        hidden_input = torch.cat((pos_prototypes, neg_prototypes), dim=1)
        logits = self.decoder(hidden_input, language, lang_lengths)
        return logits

    def get_prototypes(self, stims, labels, bootstrap=False, boot_max_size_mlt=2):
        '''
        bootstrap: whether or not to generate the positive/negative examples by
        sampling with replacement
        boot_max_size_mlt: determines the maximum number of positive or negative
        examples to have (n_pos*boot_max_size_mlt, etc.)
        '''
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

        if bootstrap:
            n_pos = labels.sum(dim=1)
            n_pos_samples = np.random.randint(1, n_pos)


        labels_mat = labels.unsqueeze(dim = 2)
        # labels: (batch_size, num_examples, 1)
        ### bmm: stands for batch matrix multiplication
        pos_prototypes = torch.bmm(stim_reps, labels_mat) # (batch_size, rep_length, 1)
        pos_prototypes = pos_prototypes.squeeze(dim = 2)
        neg_prototypes = torch.bmm(stim_reps, 1-labels_mat)
        neg_prototypes = neg_prototypes.squeeze(dim = 2)
        ### n_pos: number of positive examples of a concept (i.e. number of 1 labels)
        n_pos = labels.sum(dim = 1) # (batch_size)
        pos_prototypes = pos_prototypes / n_pos.unsqueeze(1).expand_as(pos_prototypes) # same dimension
        n_neg = (1 - labels).sum(dim = 1)
        neg_prototypes = neg_prototypes / n_neg.unsqueeze(1).expand_as(pos_prototypes)
        return pos_prototypes, neg_prototypes
        
    def sample(self, stims, labels, greedy=False):
        pos_prototypes, neg_prototypes = self.get_prototypes(stims, labels)
        hidden_input = torch.cat((pos_prototypes, neg_prototypes), dim=1)
        indices = dict(sos=self.start_index,
                       eos=self.end_index,
                       pad=self.pad_index)
        return self.decoder.sample(hidden_input, indices, greedy=greedy)

    def compute_loss(self, batch):
        """ Compute loss.
        """
        stims, labels, language, lang_lengths = self.get_inputs(batch)
        logits = self(stims, labels, language, lang_lengths)
        # logits shape: (batch size, max seq length, num vocab)
        # Assume rnn_decoder is your language model;
        # img_rep is your image representation (batch_size x hidden_size);
        # language is your list of sentences (batch_size x max_lang_length)
        # lang_length is your list of language lengths (batch_Size)
        max_seq_len = language.size(1)
        # We only care about logits up to the last token 
        # (after the last token, there's nothing to predict!)
        ### Also, we don't need to care about how it predicted the first token, 
        ### since it's always an SOS
        logits = logits[:, :-1].contiguous()
        language = language[:, 1:].contiguous()

        # Get the batch size (and make sure it's the same for all data)
        batch_size = stims.shape[0]
        assert(batch_size == language.shape[0] == lang_lengths.shape[0] == labels.shape[0])
        # "Unfold" the sequence so we have a 2d matrix
        logits_2d = logits.view(batch_size * (max_seq_len - 1), -1)
        ### TODO we probably don't need to convert language to longs here
        ### (it should already be longs, since we never converted to floats
        language_1d = language.long().view(batch_size * (max_seq_len - 1))

        # Cross entrops is your loss function - in short, you pay a penalty if you put probability amss onl
        # Note this works *without* having to normalize the softmax output
        loss = F.cross_entropy(logits_2d, language_1d, reduction='none')
        loss = loss.view(batch_size, (max_seq_len - 1))

        # Mask out losses for pad tokens
        loss *= (language != self.pad_index).float()
        # Sum up the loss for each token prediction to get a total loss per language
        total_losses = torch.sum(loss, dim=1)
        # Normalize total loss for each sequence by its length
        total_losses /= lang_lengths.float()
        # then average across language in the batch
        average_loss = torch.mean(total_losses)
        return average_loss, logits


### HELPER METHODS ###
    def get_inputs(self, batch):
        '''
        Processes input of form 
            text: (language, lengths)
            labels: length num_stimuli binary string of ground truth labels
            0-49: length num_features binary strings describing each stimulus
        into form
            language: max_language_length string of indices
            lengths: int
            stims: (num_stimuli x num_features) tensor of all stimuli
            labels
        '''
        (language, lang_lengths) = batch.text
        ### language: (batch_size, max language length)
        ### lang_lengths: (batch_size)
        if self.embeddings == "elmo":
            ### TODO change vars
            x_l_reversed = vocab_field.reverse(x_l.data)
            x_l = convert_to_elmo_ids(x_l_reversed, args.cuda)
            x_l_lengths = None
        ### get the (batch_size) tensor (basically list) of stimuli
        stims = self.construct_stim_reps(batch)
        labels = batch.__dict__['labels'].float()
        if self.cuda:
            stims = stims.to('cuda')
            language = language.to('cuda')
            lang_lengths = lang_lengths.to('cuda')
            labels = labels.to('cuda')
        return stims, labels, language, lang_lengths

    def construct_stim_reps(self, batch):
        """
        Concats separate (num_features) stimulus tensors into one 
        (num_stimuli x num_features) tensor
        """
        ### Appends all of the different features together into one column
        vals = []
        num_examples = batch.__dict__['labels'].shape[1]
        for i in range(num_examples):
            vals.append(batch.__dict__[str(i)].unsqueeze(dim=1)) ### (batch_size, 1)
        stims = torch.cat(vals, dim=1)
        ### convert entire tensor into floats
        return stims.float()


