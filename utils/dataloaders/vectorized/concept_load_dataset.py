"""
File: concept_load_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 22, 2019
Description: Load dataset using TorchText.
"""
import os
import re
import pandas as pd
import spacy

import torch
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
import torchtext.data as data
from allennlp.modules.elmo import batch_to_ids

from utils.constants import Constants

# Spacy Tokenizer
nlp = spacy.load('en_core_web_sm')

# ------------------
# Tokenization Class
# ------------------
def tokenize_fct(text):
    return [tok.text for tok in nlp.tokenizer(text)]

def tokenize_fct_lemmatize(text):
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
            tokenizer = tokenize_fct
        return SplitReversibleField(
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


### Note: we're using torchtext for its torchtext.data ("data") library
### Apparently torchtext is a "text preprocessing" library, designed to work with any DL library
### http://anie.me/On-Torchtext/ <----- excellent tutorial
### General goal: translate sentences into lists of indices that index into word embeddings?
### Q: why do we use .tsv here? Is it just an ML-wide standard? 
def load_dataset(file_template, lemmatized=False):
    """ Read text components of dataset into memory and preprocess accordingly.
    @param file_template: './data/xsd/{}/data.tsv' -- file path except for train/val/test 
    """
    # Define TorchText fields
    ### these are still just strings pointing NOT to folders, but rather .tsv's in folders
    train_file = file_template.format('train')
    val_file = file_template.format('val')
    test_file = file_template.format('test')

    ### Read in the labels of the columns as a list
    columns = pd.read_csv(test_file, sep='\t').columns.values.tolist()
    ### Dict of fields (torchtext types for columns of tabular dataset, implying how to preprocess them)
    column_field_types = {}
    ### List of feature fields, which are later to be turned into embedded vectors
    stim_fields = []
    for c in columns:
        if '-' in c:
            column_field_types[c] = (c, construct_field('numeric_label'))
            stim_fields.append(c)
        elif c in ['stim_num', 'rule_idx']:
             column_field_types[c] = (c, construct_field('numeric_label'))           
        elif c in ['true_label', 'teacher_label', 'student_label']:
            column_field_types[c] = (c, construct_field('bool_label'))
        elif c == 'text':
            column_field_types[c] = (c, construct_field('input_text', lemmatized=lemmatized))
            
    train = TabularDataset(train_file, format='tsv', fields=column_field_types)
    val = TabularDataset(val_file, format='tsv', fields=column_field_types)
    test = TabularDataset(val_file, format='tsv', fields=column_field_types)

    return train, val, test, column_field_types, stim_fields
    

def gen_text_preprocessor():
    """ Text field preprocessor for TorchText.
    """
    def clean_str(string):
        # Replace multiple spaces with a single space.
        string = re.sub(r'\s+', ' ', string).strip()

        # Replace creature names with "creature"
        creature_regexes = [
            r'kwep(s)?',
            r'morseth(s)?',
            r'luzak(s)?',
            r'zorb(s)?',
            r'oller(s)?',
        ]
        creature_misspellings = [
            r'kweep(s)?',
            r'kewps(s)?',
            r'kweb(s)?',
            r'luzek(s)?',
            r'kewp(s)?',
            r'kewpt(s)?',
            r'kwerp(s)?',
            r'lulaz(s)?',
            r'lusak(s)?',
            r'moreseth(s)?',
            r'moresth(s)?',
            r'morthess(es)?'
            r'moseth(s)?'
        ]
        for expr in creature_regexes:
            string = re.sub(expr, 'creature', string)
        for expr in creature_misspellings:
            string = re.sub(expr, 'creature', string)

        # Replace '(' with ' '
        string = re.sub(r'\(', ' ', string)
        string = re.sub(r'"+', '', string)
        return string

    return data.Pipeline(clean_str)


def construct_stim_reps(batch, stim_fields):
    """ Construct stimulus representations for batch.
    """
    vals = []
    for f in stim_fields:
        vals.append(batch.__dict__[f].unsqueeze(dim=1)) # (1, batch_size)
    stims = torch.cat(vals, dim=1)
    return stims.float()


def construct_y(batch, train_obj):
    """ Constructs tensor of labels for training the model.
    """
    if train_obj == Constants.TRAIN_OBJECTIVES['ground_truth']:
        return batch.true_label
    elif train_obj == Constants.TRAIN_OBJECTIVES['teacher']:
        return batch.teacher_label
    elif train_obj == Constants.TRAIN_OBJECTIVES['student']:
        return batch.student_label
    else:
        raise Exception('Invalid train object')


def convert_to_elmo_ids(batch_text, cuda):
    """ Convert batch of text into word ids / char ids for ELMO.
    """
    elmo_x = [x.split(' ') for x in batch_text]
    elmo_x = sorted(elmo_x, key=len, reverse=True)
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
