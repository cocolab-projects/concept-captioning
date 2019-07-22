"""
File: construct_vectorized_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 20, 2019
Description: Process data.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from shutil import copyfile
import spacy
from sklearn.model_selection import train_test_split

# ----------------
# PROCESS RAW DATA
# ----------------

# --------------
# REFERENCE GAME
# --------------
def referenceSetup(data_dir):
    """ Read raw data into memory (messages, responses, and stimuli summaries).
        Then split this into train/val/test splits.
    """
    msgs, responses, _, stims = read_raw_data('./data/reference/{}/raw/chatMessage'.format(data_dir), './data/reference/{}/raw/clickedObj'.format(data_dir), None, './data/reference/{}/raw/stimuli'.format(data_dir), True)
    construct_reference_splits(msgs, responses, stims, data_dir, train=90, val=5, test=5)


def construct_reference_splits(msgs, responses, stims, data_dir, train=70, val=15, test=15):
    """ Split concepts into train/val/test splits.
    """
    assert (train + val + test == 100), "Train, Val, Test should add up to 100%. It currently does not."

    # Assign example identifiers
    msgs['example_id'] = msgs.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
    responses['example_id'] = responses.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
    stims['example_id'] = stims.apply (lambda row: hash(str(row['trialNum']) + row['gameid']), axis=1)
    exampleIds = list(set(msgs['example_id'].tolist()))

    # Identify train, val, and test rules
    trainExampleIds, other = train_test_split(exampleIds, train_size=(train/100.0), random_state=42)
    valExampleIds, testExampleIds = train_test_split(other, train_size=(val/(1.0 * (val + test))), random_state=42)

    # Create necessary directories
    splits = ['train', 'val', 'test']
    for split in splits:
        dir = './data/reference/{}/{}'.format(data_dir, split)
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Split messages
    trainMsgs = msgs.loc[msgs['example_id'].isin(trainExampleIds)]
    trainMsgs.to_csv('./data/reference/{}/train/msgs.tsv'.format(data_dir), sep='\t', index=False)
    valMsgs = msgs.loc[msgs['example_id'].isin(valExampleIds)]
    valMsgs.to_csv('./data/reference/{}/val/msgs.tsv'.format(data_dir), sep='\t', index=False)
    testMsgs = msgs.loc[msgs['example_id'].isin(testExampleIds)]
    testMsgs.to_csv('./data/reference/{}/test/msgs.tsv'.format(data_dir), sep='\t', index=False)
    
    # Split Responses
    trainResponses = responses.loc[responses['example_id'].isin(trainExampleIds)]
    trainResponses.to_csv('./data/reference/{}/train/responses.tsv'.format(data_dir), sep='\t', index=False)
    valResponses = responses.loc[responses['example_id'].isin(valExampleIds)]
    valResponses.to_csv('./data/reference/{}/val/responses.tsv'.format(data_dir), sep='\t', index=False)
    testResponses = responses.loc[responses['example_id'].isin(testExampleIds)]
    testResponses.to_csv('./data/reference/{}/test/responses.tsv'.format(data_dir), sep='\t', index=False)
    
    # Split stimuli
    trainStims = stims.loc[stims['example_id'].isin(trainExampleIds)]
    trainStims.to_csv('./data/reference/{}/train/stims.tsv'.format(data_dir), sep='\t', index=False)
    valStims = stims.loc[stims['example_id'].isin(valExampleIds)]
    valStims.to_csv('./data/reference/{}/val/stims.tsv'.format(data_dir), sep='\t', index=False)
    testStims = stims.loc[stims['example_id'].isin(testExampleIds)]
    testStims.to_csv('./data/reference/{}/test/stims.tsv'.format(data_dir), sep='\t', index=False)


def read_raw_data(chatMsgsDir=None, testResponsesDir=None, stimsSummaryFilePath=None, stimsDir=None, refGames=False):
    def read_chat_messages(dir):
        """ Read chat messages into memory.
        """
        if not os.path.exists(dir):
            raise Exception('No directory {} exists for chat messages'.format(dir))
        msgFiles = glob.glob(os.path.join(dir, "*.tsv"))
        msgs = pd.concat((pd.read_csv(f, sep='\t') for f in msgFiles))
        return msgs

    def read_test_responses(dir):
        """ Read test-time responses into memory.
        """
        if not os.path.exists(dir):
            raise Exception('No directory {} exists for test responses'.format(dir))
        responseFiles = glob.glob(os.path.join(dir, "*.csv"))
        responses = pd.concat((pd.read_csv(f, sep='\t') for f in responseFiles))
        return responses

    def read_stims_summary(fp):
        """ Read stimuli summary into memory.
        """
        if not os.path.exists(fp):
            raise Exception('No file {} exists for test stimuli'.format(fp))
        stimsSummary = pd.read_json(fp).transpose()
        return stimsSummary

    def read_stims(dir):
        """ Read stimuli into memory.
        """
        if refGames == False:
            raise Exception('Reading stimuli non-reference games format, has not been implemented')
        else:
            if not os.path.exists(dir):
                raise Exception('No directory {} exists for test stimuli'.format(dir))
            stimFiles = glob.glob(os.path.join(dir, "*.json"))
            stimData = []
            for stimFile in stimFiles:
                with open(stimFile) as f:
                    data = json.load(f)
                    for round, roundData in data.items():
                        d = json_normalize(roundData)
                        gameid = os.path.splitext(os.path.split(stimFile)[1])[0]
                        d['gameid'] = gameid
                        d['trialNum'] = round
                        stimData.append(d)
            stims = pd.concat(stimData)
            return stims

    msgs, responses, stimsSummmary, stims = None, None, None, None
    if chatMsgsDir is not None:
        msgs = read_chat_messages(chatMsgsDir)
    if testResponsesDir is not None:
        responses = read_test_responses(testResponsesDir)
    if stimsSummaryFilePath is not None:
        stimsSummmary = read_stims_summary(stimsSummaryFilePath)
    if stimsDir is not None:
        stims = read_stims(stimsDir)
    return msgs, responses, stimsSummmary, stims

# ---------------------
# CONCEPT LEARNING GAME
# ---------------------
def conceptSetup():
    """ Read raw data into memory (messages, responses, and stimuli summaries).
        Then split this into train/val/test splits.
    """
    msgs, responses, stimsSummmary, _ = read_raw_data('./data/concept/raw/exp/chatMessage', './data/concept/raw/exp/logTest', './data/concept/raw/stims/concept_summary.json')
    construct_concept_splits(msgs, responses, stimsSummmary, './data/concept/raw/stims/test_stim/vectorized', False)


def construct_concept_splits(msgs, responses, stimsSummary, stimsDir, train=70, val=15, test=15, new_splits=False):
    """ Split concepts into train/val/test splits.
        Construct appropriate directories and write a 
        "concept_summary" file for each split.
    """
    if new_splits:
        assert (train + val + test == 100), "Train, Val, Test should add up to 100%. It currently does not."

        # Identify train, val, and test rules
        trainRules, other = train_test_split(stimsSummary, train_size=(train/100.0), random_state=42)
        valRules, testRules = train_test_split(other, train_size=(val/(1.0 * (val + test))), random_state=42)

        # Create necessary directories
        splits = ['train', 'val', 'test']
        for split in splits:
            dir_1 = './data/concept/{}/'.format(split)
            if not os.path.exists(dir_1):
                os.makedirs(dir_1)
            dir_2 = './data/concept/{}/stims'.format(split)
            if not os.path.exists(dir_2):
                os.makedirs(dir_2)

        # Write stim summaries
        trainRules.transpose().to_json('./data/concept/train/rules.json')
        valRules.transpose().to_json('./data/concept/val/rules.json')
        testRules.transpose().to_json('./data/concept/test/rules.json')
    else:
        trainRules = pd.read_json('./data/concept/train/rules.json').transpose()
        valRules = pd.read_json('./data/concept/val/rules.json').transpose()
        testRules = pd.read_json('./data/concept/test/rules.json').transpose()

    # Split messages
    trainMsgs = msgs.loc[msgs['rule_idx'].isin(trainRules.index.tolist())]
    trainMsgs.to_csv('./data/concept/train/msgs.tsv', sep='\t', index=False)
    valMsgs = msgs.loc[msgs['rule_idx'].isin(valRules.index.tolist())]
    valMsgs.to_csv('./data/concept/val/msgs.tsv', sep='\t', index=False)
    testMsgs = msgs.loc[msgs['rule_idx'].isin(testRules.index.tolist())]
    testMsgs.to_csv('./data/concept/test/msgs.tsv', sep='\t', index=False)
    
    # Split responses
    trainResponses = responses.loc[responses['rule_idx'].isin(trainRules.index.tolist())]
    trainResponses.to_csv('./data/concept/train/responses.tsv', sep='\t', index=False)
    valResponses = responses.loc[responses['rule_idx'].isin(valRules.index.tolist())]
    valResponses.to_csv('./data/concept/val/responses.tsv', sep='\t', index=False)
    testResponses = responses.loc[responses['rule_idx'].isin(testRules.index.tolist())]
    testResponses.to_csv('./data/concept/test/responses.tsv', sep='\t', index=False)

    # Split stimuli
    stimFiles = glob.glob(os.path.join(stimsDir, "*.json"))
    trainFp, valFp, testFp = trainRules.name.unique().tolist(), valRules.name.unique().tolist(), testRules.name.unique().tolist()
    for f in stimFiles:
        fn = os.path.splitext(os.path.basename(f))[0]
        if fn in trainFp:
            dst = os.path.join('./data/concept/train/vectorized/stims', os.path.basename(f))
            copyfile(f, dst)
        elif fn in valFp:
            dst = os.path.join('./data/concept/val/vectorized/stims', os.path.basename(f))
            copyfile(f, dst)
        elif fn in testFp:
            dst = os.path.join('./data/concept/test/vectorized/stims',os.path.basename(f))
            copyfile(f, dst)
        else:
            raise Exception ("Unassigned rule.")

# -----------------------
# Langauge Pre-Processors
# -----------------------

def gen_text_preprocessor(type):
    """ Return a text pre-processor function depending on the 
        desired type of preprocessing.
    """
    def concat_informative_msgs(msgs, refGames=False):
        informative_msgs = msgs[msgs.messageType == 'I']
        if refGames:
            group = informative_msgs.groupby(['example_id', 'messageType'])
            concat_msgs = pd.DataFrame(group['message'].apply(' '.join))
        else:
            group = informative_msgs.groupby(['gameid', 'rule_idx', 'messageType'])
            concat_msgs = pd.DataFrame(group['text'].apply(' '.join))
        concat_msgs = concat_msgs.reset_index()
        return concat_msgs

    def all_msgs(msgs, refGames=False, speaker_only=False):
        if refGames:
            if speaker_only:
                msgs = msgs[msgs['sender'] == 'speaker']
            group = msgs.groupby(['example_id'])
            ### TODO concat with different char than just a space?
            ### Like a period and space
            concat_msgs = pd.DataFrame(group['message'].apply(' '.join))
        concat_msgs = concat_msgs.reset_index()
        return concat_msgs
    
    if type == 'concat_informative_msgs':
        return concat_informative_msgs
    elif type == 'all_msgs':
        return all_msgs
    else:
        raise Exception('Preprocessor {} unavailable'.format(type))

def preprocess_text(dirs, type, input, output, refGames=False):
    """ Preprocess train/val/test input text files and write 
        them to disk at the output file name.
    """
    for d in dirs:
        input_file = os.path.join(d, input)
        output_file = os.path.join(d, output)
        text = pd.read_csv(input_file, sep='\t')
        preprocessor = gen_text_preprocessor(type)
        ### split up now because only all_msgs has the speaker_only option
        if type == "all_msgs":
            text_preprocessed = preprocessor(text, refGames=True, speaker_only=True)
        else:
            text_preprocessed = preprocessor(text, refGames=refGames)
        text_preprocessed.to_csv(output_file, sep='\t', index=False)


# --------------------
# DATASET CONSTRUCTION
# --------------------
class ConceptDataset():
    """ Generate datasets for concept learning experiments.
    """
    @staticmethod
    def construct_concat_informative_dataset():
        """ Construct dataset of concatenated informative message, stimulus (vectorized),
            and 3 outputs (gold truth, teacher response, student response).
        """
        dirs = ['./data/concept/train', './data/concept/val', './data/concept/test']
        preprocess_text(dirs, 'concat_informative_msgs', 'msgs.tsv', 'msgs_concat_informative.tsv')
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
                stimFilePath = os.path.join(d, 'vectorized', 'stims', '{}.json'.format(r))
                print(stimFilePath)
                s_f = pd.read_json(stimFilePath).transpose()
                s_f['stim_num'] = s_f.index
                rule_idx = rules.index[rules['name'] == r].tolist()[0]
                s_f['rule_idx'] = rule_idx
                stimuli.append(s_f)
            stimuli = pd.concat(stimuli)
            dataset = dataset.merge(stimuli)
            dataset.to_csv(os.path.join(d, 'vectorized', 'concat_informative_dataset.tsv'), sep='\t', index=False)


class ReferenceDataset():
    """ Generate datasets for Reference games.
    """
    @staticmethod
    def construct_dataset(data_dir):
        """ Construct dataset of concatenated informative message, target (vectorized),
            distractor 1 (vectorized), distractor 2 (vectorized).
            and 2 outputs (gold truth, listener selection).
        """
        dirs = ['./data/reference/{}/train'.format(data_dir), './data/reference/{}/val'.format(data_dir), './data/reference/{}/test'.format(data_dir)]
        preprocess_text(dirs, 'all_msgs', 'msgs.tsv', 'msgs_concat_speaker_only.tsv', refGames=True)
        for d in dirs:
            msgs = pd.read_csv(os.path.join(d, 'msgs_concat_speaker_only.tsv'), sep='\t')

            # Produce table of stimuli responses for a specific stimuli and conversation.
            responses = pd.read_csv(os.path.join(d, 'responses.tsv'), sep='\t')[['example_id', 'selection']]

            # Combine stimuli responses with msgs
            dataset = responses.merge(msgs)

            # Combine with stimuli representations
            stims = pd.read_csv(os.path.join(d, 'stims.tsv'), sep='\t')
            dataset = dataset.merge(stims)

            # Drop examples with incorrect answers
            dataset = dataset.loc[dataset['selection'] == 'target']
            dataset = dataset.drop(columns=['selection', 'example_id', 'gameid', 'trialNum'])
            dataset.to_csv(os.path.join(d, 'vectorized', 'dataset.tsv'), sep='\t', index=False)


def main():
    data_dir = 'pilot_coll1'
    #referenceSetup(data_dir)
    ReferenceDataset.construct_dataset(data_dir)
    # conceptSetup()
    # ConceptDataset.construct_concat_informative_dataset()


if __name__ == '__main__':
    main()
