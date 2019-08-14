'''
Author: Will Schwarzer
Date: August 14, 2019
Constructs tensors containing all targets in the reference game dataset
and all positive stimuli in the concept game dataset. To be used for creating
supplementary ref games / concept stimuli for training students.
'''

import torch
import pandas as pd
import os

DATA_DIR = "./data/train/{}/vectorized/"
CONCEPT_DSET_NAME = "concept_dataset.tsv"
REF_DSET_NAME = "ref_dataset.tsv"
OUT_NAME = "targets.pt"
feat_to_creat = torch.tensor([0]*11 + \
                             [1]*17 + \
                             [2]*12 + \
                             [3]*18 + \
                             [4]*20)

def get_pos_stims(file):
    '''
    Returns a torch tensor of all positive stimuli in the given dataset,
    as indicated by the 'labels' column. The tensor is separated into
    different animals as follows:
    0 - bird, 1 - bug, 2 - fish, 3 - flower, 4 - tree
    The dataset is assumed to be a pandas-readable tsv, where each row
    represents a set of stimuli in columns labeled 0, 1, ..., n_stims, 
    along with a column named 'labels', consisting of an n_stims-length
    binary string indicating which stims are 'positive' and which are 
    'negative'.
    '''
    games = pd.read_csv(file, sep='\t', dtype=str)
    

def write_target_file(data_dir, input_name):
    input_file = os.path.join(data_dir, input_name)
    targets = get_pos_stims(input_file)
    output = os.path.join(data_dir, OUT_NAME)
    torch.save(targets, output)

if __name__ == '__main__':
    ref_dir = DATA_DIR.format('reference')
    write_target_file(ref_dir, REF_DSET_NAME)
    concept_dir = DATA_DIR.format('concept')
    write_target_file(concept_dir, CONCEPT_DSET_NAME)
