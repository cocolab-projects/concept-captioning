"""
File: construct_vision_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 30, 2019
Description: Process data.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import re
from shutil import copyfile
import spacy
from sklearn.model_selection import train_test_split

# --------------------
# DATASET CONSTRUCTION
# --------------------
class ConceptDataset():
    """ Generate datasets for concept learning experiments.
    """
    @staticmethod
    def construct_dataset():
        """ Construct dataset of concatenated informative message, stimulus (image),
            and 3 outputs (gold truth, teacher response, student response).
        """
        dirs = ['./data/concept/train', './data/concept/val', './data/concept/test']
        for d in dirs:
            msgs = pd.read_csv(os.path.join(d, 'msgs_concat_informative.tsv'), sep='\t')
            rules = pd.read_json(os.path.join(d, 'rules.json')).transpose()

            # Produce table of stimuli responses for a specific stimuli and conversation.
            responses = pd.read_csv(os.path.join(d, 'responses.tsv'), sep='\t')[['stim_num', 'turker_label', 'true_label', 'gameid', 'role', 'rule_idx']]
            t_responses = responses[responses.role == 'explorer'].rename(columns={'turker_label': 'teacher_label'}).drop(columns=['role'])
            s_responses = responses[responses.role == 'student'].rename(columns={'turker_label': 'student_label'}).drop(columns=['role'])
            responses = t_responses.merge(s_responses)

            # Combine stimuli responses with msgs
            dataset = responses.merge(msgs)

            # Combine with stimuli representations
            stimuli = []

            for r in rules['name'].tolist():
                img_ids_fp = os.path.join(d, 'vision', 'ids', '{}.csv'.format(r))
                img_ids_df = pd.read_csv(img_ids_fp)
                img_ids = ['{}.png'.format(os.path.splitext(os.path.split(x)[1])[0]) for x in img_ids_df['id'].tolist()]
                img_ids_df['id'] = img_ids
                img_ids_df['stim_num'] = img_ids_df.index
                img_ids_df['rule_idx'] = rules.index[rules['name'] == r].tolist()[0]
                stimuli.append(img_ids_df)
            stimuli = pd.concat(stimuli)
            dataset = dataset.merge(stimuli)
            dataset.to_csv(os.path.join(d, 'vision', 'concat_informative_dataset.tsv'), sep='\t', index=False)


class ReferenceDataset():
    """ Generate datasets for Reference games.
    """
    @staticmethod
    def construct_dataset(data_dir):
        """ Construct dataset of concatenated informative message, target (image),
            distractor 1 (image), distractor 2 (image) and 2 outputs (gold truth, listener selection).
        """
        dirs = ['./data/reference/{}/train'.format(data_dir), './data/reference/{}/val'.format(data_dir), './data/reference/{}/test'.format(data_dir)]
        for d in dirs:
            msgs = pd.read_csv(os.path.join(d, 'msgs.tsv'), sep='\t')
            msgs['example_id'] = msgs.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
            group = msgs.groupby(['example_id'])
            concat_msgs = pd.DataFrame(group['message'].apply(' '.join))
            concat_msgs = concat_msgs.reset_index()

            # Produce table of stimuli responses for a specific stimuli and conversation.
            responses = pd.read_csv(os.path.join(d, 'responses.tsv'), sep='\t')[['trialNum', 'gameid', 'selection']]
            responses['example_id'] = responses.apply(lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)

            # Combine stimuli responses with msgs
            dataset = responses.merge(concat_msgs)

            # Combine with stimuli ids
            stim_ids = []
            stim_id_files = glob.glob('./data/reference/{}/raw/vision/ids/*.json'.format(data_dir))
            for f in stim_id_files:
                x = pd.read_json(f).transpose()
                x['trialNum'] = x.index
                x['gameid'] = os.path.splitext(os.path.split(f)[1])[0]
                x['example_id'] = x.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
                for c in ['distr1', 'distr2', 'target']:
                    x[c] =[re.sub(r'.svg', '.png', img_name) for img_name in x[c].tolist()]
                stim_ids.append(x)

            # Drop examples with incorrect answers
            stim_ids = pd.concat(stim_ids)
            dataset = dataset.merge(stim_ids)
            dataset = dataset.loc[dataset['selection'] == 'target']
            dataset = dataset.drop(columns=['selection', 'gameid', 'trialNum'])
            dataset.to_csv(os.path.join(d, 'vision', 'dataset.tsv'), sep='\t', index=False)


def main():
    # concept_dataset = ConceptDataset()
    # concept_dataset.construct_dataset()
    ref_dataset = ReferenceDataset()
    ref_dataset.construct_dataset('pilot_coll1')

if __name__ == '__main__':
    main()
