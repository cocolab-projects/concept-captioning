"""
File: constants.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: February 20, 2019
Description: Constants for utilities
"""

class Constants(object):
    """ Constants for utilities.
    """
    START_TOKEN = '<s>'
    END_TOKEN = '<e>'
    PAD_TOKEN = '<p>'
    UNK_TOKEN = '<unk>'

    UTT_CLASSES = {
        '?': 'question',
        'O': 'other',
        'G': 'generic',
        'A': 'yes/no',
        'Q': 'quantifier',
        'C': 'conditional',
        'I': 'imperative',
        'P': 'adverbial',
        'E': 'exemplar',
        'N': 'numeric',
        'L': 'logical',
        'EO': 'either-or',
        'Y': 'yes',
    }

    MSG_TYPES = {
        'S': 'social',
        'F': 'follow-up',
        'M': 'misc',
        'I': 'informative'
    }

    TRAIN_OBJECTIVES = {
        'ground_truth': 'ground_truth',
        'teacher': 'teacher',
        'student': 'student'
    }
