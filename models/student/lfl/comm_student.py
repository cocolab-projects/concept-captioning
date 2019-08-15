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
from utils.torch_utils import to_onehot

import torchvision.models as models

import pickle
import numpy as np

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

        self.language_model = BiLSTM(
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
            self.stim_model = MLP(
                i_dim=kwargs['i_dim_s_student'],
                h_dim=kwargs['h_dim_s_student'],
                o_dim=kwargs['o_dim_s_student'],
                d_prob=kwargs['d_prob_s_student'],
                batch_norm=kwargs['bn_student'],
                num_layers=kwargs['num_layers_s_student'],
            )
        elif kwargs['stim_model_type'] == 'resnet':
            self.stim_model = models.resnet18(pretrained=kwargs['pretrained'])
            self.stim_model.fc = nn.Linear(512, kwargs['o_dim_stim'])
        else:
            raise Exception('Invalid Stimulus Model')
            
        
        self.ablate_lang = kwargs['ablate_student_lang']
        if self.ablate_lang:
            comp_i_dim = kwargs['o_dim_s_student']
        else:
            comp_i_dim = kwargs['o_dim_l_student'] + kwargs['o_dim_s_student']
        self.comparator = MLP(
            i_dim=comp_i_dim,
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

        # Load all targets or positive stims for ref and concept games,
        # to allow train-time sampling of new ref/concept games
        with open(kwargs['ref_targets'], 'rb') as ref_targets:
            self.ref_targets = pickle.load(ref_targets)
        # TODO implement concept targets + supp game sampling
        # with open(kwargs['concept_targets'], 'rb') as concept_targets:
        #     self.concept_targets = pickle.load(concept_targets)
        self.n_supp = kwargs['n_supp_ref_games']

        # Used for translating feature reps to animal type
        # 0 - bird, 1 - bug, 2 - fish, 3 - flower, 4 - tree
        self.feat_to_creat = torch.tensor([0]*11 + \
                                          [1]*17 + \
                                          [2]*12 + \
                                          [3]*18 + \
                                          [4]*20)
        if self.cuda:
            self.feat_to_creat = self.feat_to_creat.to('cuda')

        self.is_eval = False
        self.include_supp_acc = kwargs['include_supp_acc']

    def forward(self, lang, stims, lang_lengths, onehot=False):
        """ lang: language (describing concept)
            stims: stimulus (from test set)
            lang_lengths: true lengths of text in lang
            use_concept: T/F use concept model
        """
        language_rep, alphas = self.language_model(lang, lang_lengths, onehot=onehot)
        stim_rep = self.stim_model(stims)
        joined_rep = torch.cat([language_rep, stim_rep], dim=1)
        if self.ablate_lang:
            logits = self.comparator(stim_rep)
        else:
            logits = self.comparator(joined_rep)
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
        breakpoint()
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
        logits: (batch*(self.n_supp+1), 3)
        self.labels: (batch)
        '''
        assert (logits.shape[0] == self.labels.shape[0]), \
                "Logits and labels not of the same shape!"
        if self.is_eval and not self.include_supp_acc:
            # Only compute accuracy for original games
            logits = logits.view(-1, self.n_supp+1, 3)
            logits = logits[:, 0, :]
            labels = self.labels.view(-1, self.n_supp+1)
            labels = labels[:, 0]
        else:
            labels = self.labels
        preds = logits.argmax(1)
        correct = preds == labels
        n_correct = sum(correct).item()
        n_total = logits.shape[0]
        return n_correct, n_total

    ### HELPER METHODS ###
    def get_inputs_batch(self, batch, onehot=False):
        stims = self.construct_stim_reps(batch)
        return self.get_inputs(stims, batch.labels, batch.text[0], 
                          batch.text[1], onehot=onehot)

    def get_inputs(self, stims, labels, lang, lang_lengths, onehot=False):
        if self.cuda:
            lang = lang.to('cuda')
            lang_lengths = lang_lengths.to('cuda')
            stims = stims.to('cuda')
            labels = labels.to('cuda')
        # TODO throw an error if they try to use elmo?
        max_msg_size = lang.shape[1]
        # repeat all inputs 3 times for 3 stims, once for each ref game
        n_stims = 3*(1+self.n_supp)
        if onehot:
            n_vocab = lang.shape[2]
            lang = lang.unsqueeze(1).expand(-1, n_stims, -1, -1) 
            lang = lang.contiguous().view(-1, max_msg_size, n_vocab)
        else:
            lang = lang.unsqueeze(1).expand(-1, n_stims, -1) 
            # TODO does having to use contiguous() defeat the purpose of
            # expand()?
            lang = lang.contiguous().view(-1, max_msg_size)

        lang_lengths = lang_lengths.unsqueeze(1).expand(-1, n_stims)
        lang_lengths = lang_lengths.contiguous().view(-1)

        # Flatten stims (one per row); used if ref games were generated in
        # communication game
        if len(stims.shape) > 2:
            n_feats = stims.shape[2]
            stims = stims.view(-1, n_feats)

        # Generate stims for supplementary ref games
        if self.n_supp:
            stims = self.add_supp_games(stims, labels)

        # Convert the one-hot labels into ints to use with cross-entropy
        labels = labels.argmax(1)
        # Repeat for each supplementary ref game
        labels = labels.unsqueeze(1).expand(-1, 1+self.n_supp)
        labels = labels.contiguous().view(-1)

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

    def add_supp_games(self, stims, labels, ref=True):
        '''
        Creates self.n_supp additional games for each reference game in stims,
        where each supplementary game uses the same target as the original,
        but uses distractors sampled from the *targets* of *other species*. 
        (This is to avoid the student learning that some stims appear as
        targets more than distractors.)
        These games are then catted onto the original games, and returned in 
        the form of an n_stims*(1+self.n_supp) length tensor of stims.
        stims: (batch*3, n_feats)
        '''
        assert self.n_supp, "Should create a nonzero number of supp games!"
        n_feats = stims.shape[1]
        # Get only targets by indexing into ref game stims by their labels
        # TODO this only works for ref games; need to change this line
        # and also use nonzero() instead of argmax below
        targets = stims.view(-1, 3, n_feats)
        # Get the index of the target
        labels = labels.argmax(1)
        # In order to use gather, labels has to have the same size
        # as targets
        labels_expanded = labels.unsqueeze(1).unsqueeze(2)
        labels_expanded = labels_expanded.expand(-1, 1, n_feats)
        # gather syntax: (input (to be indexed into), dim, index)
        targets = torch.gather(targets, 1, labels_expanded)
        # targets shape: (batch, 1, n_feats)
        # Repeat once for each game
        targets = targets.expand(-1, self.n_supp, -1).contiguous()
        targets = targets.view(-1, n_feats) # (batch*(n_supp), n_feats)
        # Get tensor of target types
        # Find index of last present feature, and determine which
        # animal range it's in
        target_types = targets.argmax(1)
        target_types = torch.gather(self.feat_to_creat, 0, target_types)
        target_types = to_onehot(target_types, n=5) # (batch, 5)
        # Get valid new animal types as all except the previous type
        new_distr_types = torch.ones(target_types.shape, device=target_types.device) - target_types
        # Duplicate once to get two distractors
        new_distr_types = new_distr_types.unsqueeze(1).expand(-1, 2, -1).contiguous()
        # new_distrs shape: (batch*(n_supp), 2, 5)
        new_distr_types = new_distr_types.view(-1, 5)
        # Sample from new animal types
        new_distr_types = new_distr_types.multinomial(1).squeeze() # (batch*(n_supp))
        # new_targets is now the index of the animal types to be used
        new_distrs = torch.zeros([new_distr_types.shape[0], n_feats], device=new_distr_types.device)
        # TODO convert this to torch magic
        for distr_type_idx, creat_type in enumerate(new_distr_types):
            if ref:
                n_poss = len(self.ref_targets[creat_type])
                distr_idx = np.random.randint(n_poss)
                new_distrs[distr_type_idx] = self.ref_targets[creat_type][distr_idx]
            else:
                n_poss = len(self.concept_targets[creat_type])
                distr_idx = np.random.randint(n_poss)
                new_distrs[distr_type_idx] = self.concept_targets[creat_type][distr_idx]
        # new_distrs is now actual stimulus feature representations
        # shape: (batch*2*(n_supp), n_feats)
        # Now undo the flattening of the two distractors
        new_distrs = new_distrs.view(-1, 2, n_feats) # (batch*(n_supp), ...)
        # Cat the original targets with the new distractors to produce the
        # supp games
        targets.unsqueeze_(1)
        new_games = torch.cat([targets, new_distrs], dim=1) 
        # shape: (batch*n_supp, 3, n_feats)
        # Now combine with the original games
        new_games = new_games.view(-1, 3*self.n_supp, n_feats)
        stims = stims.view(-1, 3, n_feats)
        stims = torch.cat([stims, new_games], dim=1)
        stims = stims.view(-1, n_feats) # (batch*3*(1+n_supp), n_feats)
        return stims

    def eval(self):
        super().eval()
        self.is_eval = True

    def train(self, is_train=True):
        super().train(is_train)
        self.is_eval = False
