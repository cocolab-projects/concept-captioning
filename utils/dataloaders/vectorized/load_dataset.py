"""
File: concept_load_dataset_teacher.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 8, 2019
Description: Load dataset using TorchText, such that each row is a single concept
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

def tokenize_binary_string(string):
    return [int(char) for char in string]

def construct_field(
    field_type,
    batch_first=True,    
    input_lower=True,
    lemmatized=False,
    input_include_lengths=True,
    input_fix_length=None,
):
    ### batch_first: creates tensors with batch as the first dimension 
    ### (i.e. when outputting through iterator)

    """ Construct TorchText field.

        Note: the `input_<x>` fields are specifically parameters for
              the `input_text` field type.
    """
    if field_type == 'input_text':
        ### tokenize_{}: fns defined above
        if lemmatized:
            tokenizer = tokenize_fct_lemmatize
        else:
            tokenizer = tokenize_fct
        ### a type of field defined below; used for all teacher descriptions
        ### allows tokenization to be reversed
        ### also note: we replace all creature names with 'creature' 
        ### (presumably just assuming that that info would be more useful to the nn)
        ### note: "vocab" is a mapping from words to integers
        ### the sequential option just determines whether or not the data gets tokenized
        ### note also that we're NOT padding sequences to a fixed length
        ### however, we are using initial and end tokens
        ### include_lengths = true: means that our iterator will return (minibatch, seq_lengths)
        ### (where seq_lengths is a list of the lengths of our sequences)
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
    ### Transforms student/teacher responses, and ground truth, to ints
    elif field_type == 'bool_label':
        return LabelField(
            use_vocab=False,
            batch_first=batch_first,
            preprocessing = lambda x: (x == 'True')
        )
    elif field_type == 'binary_string':
        return Field(
            use_vocab=False,
            batch_first=batch_first,
            sequential = True,
            tokenize = tokenize_binary_string
        )
    else:
        raise Exception('Invalid Field Type')

def load_dataset(file_template, lemmatized=False, text_field = None):
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
    ### Of type (str, field)
    column_field_types = {}
    if text_field is None:
        text_field = construct_field('input_text', lemmatized=lemmatized)
    for c in columns:
        if c == 'text':
            column_field_types[c] = (c, text_field)
        else:
            column_field_types[c] = (c, construct_field('binary_string'))
            
    train = TabularDataset(train_file, format='tsv', fields=column_field_types)
    val = TabularDataset(val_file, format='tsv', fields=column_field_types)
    test = TabularDataset(test_file, format='tsv', fields=column_field_types)

    # Not returning test file yet (saving that for the end of the experiment)
    return train, val, text_field

### Note: we're using torchtext for its torchtext.data ("data") library
### Apparently torchtext is a "text preprocessing" library, designed to work with any DL library
### http://anie.me/On-Torchtext/ <----- excellent tutorial
### General goal: translate sentences into lists of indices that index into word embeddings?
### Q: why do we use .tsv here? Is it just an ML-wide standard? 
def load_dataset_old(file_template, lemmatized=False):
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
    ### Of type (str, field)
    column_field_types = {}
    ### List of feature fields, which are later to be turned into embedded vectors
    ### Data structure: gameid is a unique id for a teacher/concept combination - i.e. one game
    ### So, presumably, (gameid, text) pairings are unique
    ### df_red = df[['gameid', 'text']].drop_duplicates()
    ### df_red.duplicated(subset = 'gameid') --> 10 results, but their gameids are all the same???
    ### So we should use text to group for now; but eventually should figure this out and move to using gameid
    ### TODO ask Jesse what's going on

    ### Q: Why do we need to pass this uncompressed stim_fields back here?
    ### Why can't we compress the features in this function?
    ### Maybe ttext wouldn't be able to handle multi-label columns
    ### ALso, ttext reads directly from the file, not from a pd df
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
            
    ### Why are we throwing away the test data? (answered)
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
        ### TODO this seems sketchy (people already use "creature" in their descriptions);
        ### maybe should replace with some completely unique string
        for expr in creature_regexes:
            string = re.sub(expr, 'creature', string)
        for expr in creature_misspellings:
            string = re.sub(expr, 'creature', string)

        # Replace '(' with ' '
        string = re.sub(r'\(', ' ', string)
        string = re.sub(r'"+', '', string)
        return string

    return data.Pipeline(clean_str)


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

### taken from http://anie.me/On-Torchtext/ (Or maybe written by that person directly :) )
### Used to create a field with reversible mapping and(?) tokenization
### i.e. lets you go from [5, 1, 3, 2] to "Hi, there!"
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
