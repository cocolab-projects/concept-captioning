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
import pickle

DATA_DIR = "./data/{}/train/vectorized/"
CONCEPT_DSET_NAME = "concept_dataset.tsv"
REF_DSET_NAME = "ref_dataset.tsv"
OUT_NAME = "targets.pkl"
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
    games = games[[col for col in games.columns if col != 'text']]
    for col in games.columns:
        games[col] = games[col].apply(tokenize_binary_string)
    # labels = torch.tensor(games['labels'])
    # games = games[[col for col in games.columns if col != 'labels']]
    # stims = torch.tensor([games[col] for col in games.columns])
    # positive_idx = 
    # TODO finish doing this robustly with labels and for the concept game
    # Find index of last present feature, and determine which
    # animal range it's in
    targets = torch.tensor(games['2']).byte()
    target_types = targets.argmax(1)
    target_types = torch.gather(feat_to_creat, 0, target_types)
    n_types = len(target_types.unique())
    sorted_targets = [[] for _ in range(n_types)]
    for target, type in zip(targets, target_types):
        sorted_targets[type].append(target)
    for idx, creat_list in enumerate(sorted_targets):
        creat_list[0].unsqueeze_(0)
        for creat in creat_list[1:]:
            creat_list[0] = torch.cat([creat_list[0], creat.unsqueeze(0)])
        sorted_targets[idx] = creat_list[0]
    breakpoint()
    # sorted_targets: length n_types list of (n_creats, n_feats) tensors
    return sorted_targets

def write_target_file(data_dir, input_name):
    input_file = os.path.join(data_dir, input_name)
    targets = get_pos_stims(input_file)
    with open(os.path.join(data_dir, OUT_NAME), 'wb') as output:
        pickle.dump(targets, output)
    breakpoint()

def tokenize_binary_string(s):
    return [int(char) for char in s]

if __name__ == '__main__':
    ref_dir = DATA_DIR.format(os.path.join('reference', 'pilot_coll1'))
    write_target_file(ref_dir, REF_DSET_NAME)
    # concept_dir = DATA_DIR.format('concept')
    # write_target_file(concept_dir, CONCEPT_DSET_NAME)
