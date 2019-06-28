"""
File: concept_load_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 30, 2019
Description: Load dataset using TorchText.
"""
import dill
import logging
import os
import pandas as pd
from PIL import Image
import re
import spacy

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.data import Field, LabelField, ReversibleField
from torchtext.data import TabularDataset
from torchvision import transforms

from utils.constants import Constants

# -------
# Globals
# -------
nlp = spacy.load('en')
logging.getLogger().setLevel(logging.INFO)


class ConceptDatasetException(Exception):
    pass


class ConceptDataset(Dataset):
    """ Dataloader for the Cultural Ratchet Concept Learning Dataset.
    """
    def __init__(self, data_dir, split, dataset_file='concat_informative_dataset.tsv',
        lemmatized=False, image_size=224
    ):
        super(ConceptDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.lemmatized = lemmatized
        self.image_size = image_size

        # Identify dataset store (labels, text, etc.) and images store
        self.dataset_store = os.path.join(self.data_dir, split, 'vision', dataset_file)
        self.images_store = os.path.join(self.data_dir, split, 'vision', 'imgs')
        if not os.path.isfile(self.dataset_store):
            raise ConceptDatasetException('Dataset File Could Not Be Found: {}'.format(self.dataset_store))
        if not os.path.isdir(self.images_store):
            raise ConceptDatasetException('Images Store Could Not Be Found: {}'.format(self.images_store))

        # Construct Data Fields
        data = pd.read_csv(self.dataset_store, sep='\t')
        self.fields = {}
        for c in data.columns.values.tolist():
            if c in ['stim_num', 'rule_idx', 'gameid']:
                self.fields[c] = (c, construct_field('numeric_label'))           
            elif c in ['true_label', 'teacher_label', 'student_label']:
                self.fields[c] = (c, construct_field('bool_label'))
            elif c == 'text':
                self.fields[c] = (c, construct_field('input_text', lemmatized=lemmatized))
            elif c in ['id', 'messageType']:
                pass
            else:
                raise ConceptDatasetException("Invalid column name found in datset: {}".format(c))

        # Images
        # Normalization Per: https://pytorch.org/docs/master/torchvision/models.html
        self.img_ids = data['id'].tolist()
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        ])

        # Build Vocab
        logging.info("Building new {} vocab...".format(split))
        self.dataset = TabularDataset(self.dataset_store, format='tsv', fields=self.fields)
        self.fields['text'][1].build_vocab(self.dataset, vectors="glove.840B.300d")


    def __getitem__(self, item):
        example = self.dataset[item]

        # Retrieve Image
        img_fp = os.path.join(self.images_store, self.img_ids[item])
        img = Image.open(img_fp).convert('RGB')
        img = self.img_transform(img)

        return {
            'image': img,
            **example.__dict__,
        }


    def __len__(self):
        return len(self.dataset)

# ----------------------------
# Concept Game Custom DataLoader
# ----------------------------

def get_data_loader(data_dir, split, vocab_file_path=None, dataset_file='concat_informative_dataset.tsv',
    lemmatized=False, image_size=224, batch_size=32, shuffle=True, num_workers=4,
):
    concept_dataset = ConceptDataset(data_dir, split, dataset_file, lemmatized, image_size)
    vocab_field = concept_dataset.fields['text'][1]
    if vocab_file_path is not None:
        torch.save(vocab_field, vocab_file_path, pickle_module=dill)
    data_loader = DataLoader(dataset=concept_dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=curried_collate_fn(concept_dataset, vocab_field),
                            pin_memory=True)  
    return data_loader, vocab_field



def curried_collate_fn(data, vocab_field):
    """ A curried collate_fn that allows us to pass in the vocab_field so that
        we can numericalize inputs during the collation procedure.
    """
    def collate_fn(data):
        """Creates mini-batch tensors from the list of dictionaries.
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of dictionaries {}. 
                - image: torch tensor of shape (3, 224, 224).
                - stim_num: int
                - rule_idx: int
                - student_label: bool
                - teacher_label: bool
                - true_label: bool
                - text: list of strings
                - gameid: str

        Returns:
            images: torch tensor of shape (batch_size, 3, 224, 224).
            (texts, text_lengths): texts and text lengths
            student_labels: torch tensor of shape (batch_size).
            teacher_labels: torch tensor of shape (batch_size).
            true_labels: torch tensor of shape (batch_size).
            data_ids: list of (gameid, rule_idx) tuples by which one can identify the datapoints.
            
        
        Note: Batch is reverse sorted according to text lengths.
        """
        # First group fields across batches
        images_batch, texts_batch, student_labels_batch, teacher_labels_batch, true_labels_batch, gameids_batch, rule_indices_batch = [], [], [], [], [], [], []
        for d in data:
            images_batch.append(d['image'])
            student_labels_batch.append(d['student_label'])
            teacher_labels_batch.append(d['teacher_label'])
            true_labels_batch.append(d['true_label'])
            texts_batch.append(d['text'])
            gameids_batch.append(d['gameid'])
            rule_indices_batch.append(d['rule_idx'])
            
        # Next numericalize the batch and reverse sort (descending order) by
        # text lengths
        texts_batch_numericalized, texts_batch_length = vocab_field.process(texts_batch)
        data = list(zip(
            images_batch, texts_batch_numericalized, texts_batch_length, student_labels_batch, teacher_labels_batch, true_labels_batch, gameids_batch, rule_indices_batch
        ))
        data.sort(key=lambda x: x[2], reverse=True)
        images, texts, text_lengths, student_labels, teacher_labels, true_labels, gameids, rule_indices = zip(*data)
        data_ids = list(zip(gameids, rule_indices))

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge texts into a single 2D tensor.
        texts = torch.stack(texts, 0)
        text_lengths = torch.stack(text_lengths, 0)

        # Create the label tensors
        student_labels = torch.tensor(student_labels, dtype=torch.long)
        teacher_labels = torch.tensor(teacher_labels, dtype=torch.long)
        true_labels = torch.tensor(true_labels, dtype=torch.long)
        return images, (texts, text_lengths), student_labels, teacher_labels, true_labels, data_ids

    return collate_fn

# ----------------------------
# TorchText Field Construction
# ----------------------------
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


# ------------------
# Text Preprocessing
# ------------------
def tokenize_fct(text):
    return [tok.text for tok in nlp.tokenizer(text)]

def tokenize_fct_lemmatize(text):
    return [tok.lemma_ for tok in nlp.tokenizer(text)]  


def gen_text_preprocessor():
    """ Text field preprocessor for TorchText.
    """
    def clean_str(string):
        # Replace multiple spaces with a single space.
        string = re.sub(r'\s+', ' ', string).strip()

        # Replace creature names with "creature"
        creature_regexes = [
            r'kwep\S*',
            r'morseth\S*',
            r'luzak\S*',
            r'zorb\S*',
            r'oller\S*',
        ]
        creature_misspellings = [
            r'kweep\S*',
            r'kewps\S*',
            r'kweb\S*',
            r'luzek\S*',
            r'kewp\S*',
            r'kewpt\S*',
            r'kwerp\S*',
            r'lulaz\S*',
            r'lusak\S*',
            r'moreseth\S*',
            r'moresth\S*',
            r'morthess(es)?'
            r'moseth\S*'
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
