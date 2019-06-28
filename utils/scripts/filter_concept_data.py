"""
File: filter_concept_data.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 25, 2019
Description: Filter concept data according to list.
"""
import json
import os
import numpy as np
import pandas as pd

def read_dataset(data_file_template='./data/concept/{}/concat_informative_dataset.tsv', split='train'):
    """ Read dataset.
    """
    fp = data_file_template.format(split)
    return pd.read_csv(fp, sep='\t')


def read_shortlist(shortlist_file_template='./data/concept/{}/{}_teacher_student_upper_75.csv', split='train'):
    """ Read shortlist.
    """
    fp = shortlist_file_template.format(split, split)
    return pd.read_csv(fp)


def filter_dataset(
    data_file_template='./data/concept/{}/concat_informative_dataset.tsv',
    shortlist_file_template='./data/concept/{}/{}_teacher_student_upper.csv',
    filtered_data_file_template='./data/concept/{}/concat_informative_filtered_dataset_75.tsv',
    split='train'
):
    """ Filter dataset according to shortlist.
    """
    dataset = read_dataset(data_file_template, split)
    shortlist = read_shortlist(shortlist_file_template, split)
    shortlist = [tuple(x) for x in shortlist.values]
    filtered_datset = pd.concat(
        dataset[(dataset['gameid'] == gameid) & (dataset['rule_idx'] == rule_idx)] for gameid, _, rule_idx in shortlist
    )
    filtered_datset.to_csv(filtered_data_file_template.format(split), sep='\t', index=False)


if __name__ == '__main__':
    filter_dataset(split='train')
    filter_dataset(split='val')
    filter_dataset(split='test')
