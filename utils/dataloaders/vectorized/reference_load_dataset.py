"""
File: reference_load_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 7, 2019
Description: Load dataset using TorchText.
"""
import os
import re
import pandas as pd
import spacy

import torch
from torchtext.data import Field, LabelField, ReversibleField
from torchtext.data import TabularDataset
import torchtext.data as data
from allennlp.modules.elmo import batch_to_ids

from utils.constants import Constants


# ------------------
# Tokenization Class
# ------------------
def tokenize_fct_lemmatize(text):
    nlp = spacy.load('en')
    return [tok.lemma_ for tok in nlp.tokenizer(text)]  


def construct_field(
    field_type,
    batch_first=True,    
    input_lower=True,
    lemmatized=False,
    input_include_lengths=True,
    input_fix_length=None,
):
    """ Construct TorchText field.

        Note: the `input_<x>` fields are specifically parameters for
              the `input_text` field type.
    """
    if field_type == 'input_text':
        if lemmatized:
            tokenizer = tokenize_fct_lemmatize
        else:
            tokenizer = "spacy"
        return ReversibleField(
            sequential=True,
            use_vocab=True,
            init_token=Constants.START_TOKEN,
            eos_token=Constants.END_TOKEN,
            lower=input_lower,
            tokenize=tokenizer,
            batch_first=batch_first,
            pad_token=Constants.PAD_TOKEN,
            unk_token=Constants.UNK_TOKEN,
            include_lengths=input_include_lengths,
            fix_length=input_fix_length,
            preprocessing=gen_text_preprocessor()
        )        
    elif field_type == 'numeric_label':
        return LabelField(
            use_vocab=False,
            batch_first=batch_first,
        )
    elif field_type == 'bool_label':
        return LabelField(
            use_vocab=False,
            batch_first=batch_first,
            preprocessing = lambda x: (x == 'True')
        )
    else:
        raise Exception('Invalid Field Type')


def load_dataset(file_template, lemmatized=False):
    """ Read text components of dataset into memory and preprocess accordingly.
    @param file_template: './data/xsd/{}/data.tsv' -- file path except for train/val/test 
    """
    # Define TorchText fields
    train_file = file_template.format('train')
    val_file = file_template.format('val')
    test_file = file_template.format('test')

    columns = pd.read_csv(test_file, sep='\t').columns.values.tolist()
    column_field_types = {}
    stim_fields = {
        'distr1': [],
        'distr2': [],
        'target': [],
    }
    for c in columns:
        if '-' in c:
            column_field_types[c] = (c, construct_field('numeric_label'))
            if 'distr1' in c:
                stim_fields['distr1'].append(c)
            elif 'distr2' in c:
                stim_fields['distr2'].append(c)
            else:
                stim_fields['target'].append(c)
        elif c == 'message':
            column_field_types[c] = (c, construct_field('input_text'))
            
    train = TabularDataset(train_file, format='tsv', fields=column_field_types)
    val = TabularDataset(val_file, format='tsv', fields=column_field_types)
    test = TabularDataset(val_file, format='tsv', fields=column_field_types)

    return train, val, test, column_field_types, stim_fields
    

def gen_text_preprocessor():
    """ Text field preprocessor for TorchText.
    """
    def clean_str(string):
        misspellings = {
            r'pur ': 'purple',
            r'fea-': 'feather',
            r'wh-': 'white',
            r'whie': 'white',
            r'wh ': 'white',
            r'or ': 'orange',
            r'or-': 'orange',
            r'orge': 'orange',
            r'winngs': 'wings',
            r'feathes': 'feathers',
        }

        for expr, subst in misspellings.items():
            string = re.sub(expr, subst, string)

        # Replace '(' with ' '
        string = re.sub(r'\(', ' ', string)
        string = re.sub(r',', ' ', string)
        string = re.sub(r'-', ' ', string)
        string = re.sub(r'~+', ' ', string)

        # Replace multiple spaces with a single space.
        string = re.sub(r'\s+', ' ', string).strip()

        string = re.sub(r'"+', '', string)
        return string

    return data.Pipeline(clean_str)


def construct_y(batch_size, cuda=False):
    """ Constructs tensor of labels for training the model.
    """
    x = torch.zeros(batch_size)
    if cuda:
        return x.long().to('cuda')
    else:
    	return x.long()


def construct_stim_reps(batch, stim_fields):
    """ Construct stimulus representations for batch of dimension
        (3 x batch dimension, stim representation).
        target, distr1, distr2
    """
    target = torch.cat(
        [batch.__dict__[f].unsqueeze(dim=1) for f in stim_fields['target']],
        dim=1
    ).float()
    distr1 = torch.cat(
        [batch.__dict__[f].unsqueeze(dim=1) for f in stim_fields['distr1']],
        dim=1
    ).float()
    distr2 = torch.cat(
        [batch.__dict__[f].unsqueeze(dim=1) for f in stim_fields['distr2']],
        dim=1
    ).float()
    stims = torch.cat([target, distr1, distr2], dim=1)
    return stims.view(-1, 78)


def convert_to_elmo_ids(batch_text, cuda):
    """ Convert batch of text into word ids / char ids for ELMO.
    """
    elmo_x = [x.split(' ') for x in batch_text]
    elmo_x_ids = batch_to_ids(elmo_x)
    if cuda:
        return elmo_x_ids.to('cuda')
    else:
        return elmo_x_ids


class SplitReversibleField(Field):
    def __init__(self, **kwargs):
        super(SplitReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        return [' '.join(ex) for ex in batch]
