"""
File: reference_load_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: May 5, 2019
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


class RefDatasetException(Exception):
    pass


class RefDataset(Dataset):
    """ Dataloader for the Cultural Ratchet Reference Game Dataset.
    """
    def __init__(self, data_dir, split, dataset_file='dataset.tsv',
        lemmatized=False, image_size=224
    ):
        super(RefDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.lemmatized = lemmatized
        self.image_size = image_size

        # Identify dataset store (labels, text, etc.) and images store
        self.dataset_store = os.path.join(self.data_dir, split, 'vision', dataset_file)
        self.images_store = os.path.join(self.data_dir, split, 'vision', 'imgs')
        if not os.path.isfile(self.dataset_store):
            raise RefDatasetException('Dataset File Could Not Be Found: {}'.format(self.dataset_store))
        if not os.path.isdir(self.images_store):
            raise RefDatasetException('Images Store Could Not Be Found: {}'.format(self.images_store))

        # Construct Data Fields
        data = pd.read_csv(self.dataset_store, sep='\t')
        self.fields = {}
        for c in data.columns.values.tolist():
            if c == 'message':
                self.fields[c] = (c, construct_field('input_text', lemmatized=lemmatized))
            elif c in ['distr1', 'distr2', 'target']:
                pass
            elif c == 'example_id':
                self.fields[c] = (c, construct_field('numeric_label'))
            else:
                raise RefDatasetException("Invalid column name found in datset: {}".format(c))

        # Images
        # Normalization Per: https://pytorch.org/docs/master/torchvision/models.html
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        ])
        self.distr1_ids = data['distr1'].tolist()
        self.distr2_ids = data['distr2'].tolist()
        self.target_ids = data['target'].tolist()

        # Build Vocab
        logging.info("Building new {} vocab...".format(split))
        self.dataset = TabularDataset(self.dataset_store, format='tsv', fields=self.fields)
        self.fields['message'][1].build_vocab(self.dataset, vectors="glove.840B.300d")


    def __getitem__(self, item):
        example = self.dataset[item]

        # Retrieve Images
        img_fp = os.path.join(self.images_store, self.distr1_ids[item])
        img = Image.open(img_fp).convert('RGB')
        distr1_img = self.img_transform(img)

        img_fp = os.path.join(self.images_store, self.distr2_ids[item])
        img = Image.open(img_fp).convert('RGB')
        distr2_img = self.img_transform(img)

        img_fp = os.path.join(self.images_store, self.target_ids[item])
        img = Image.open(img_fp).convert('RGB')
        target_img = self.img_transform(img)       

        return {
            'distr1': distr1_img,
            'distr2': distr2_img,
            'target': target_img,
            **example.__dict__,
        }


    def __len__(self):
        return len(self.dataset)

# ----------------------------
#  Custom DataLoader
# ----------------------------

def get_data_loader(data_dir, split, vocab_file_path=None, dataset_file='dataset.tsv',
    lemmatized=False, image_size=224, batch_size=32, shuffle=True, num_workers=4,
):
    ref_dataset = RefDataset(data_dir, split, dataset_file, lemmatized, image_size)
    vocab_field = ref_dataset.fields['message'][1]
    if vocab_file_path is not None:
        torch.save(vocab_field, vocab_file_path, pickle_module=dill)
    data_loader = DataLoader(dataset=ref_dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=curried_collate_fn(ref_dataset, vocab_field),
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
                - distr1: torch tensor of shape (3, 224, 224).
                - distr2: torch tensor of shape (3, 224, 224).
                - target: torch tensor of shape (3, 224, 224).
                - message: list of strings
                - example_id: example id

        Returns:
            stims: torch tensor of shape (batch_size * 3, 3, 224, 224).
            (texts, text_lengths): texts and text lengths
            labels: torch tensor of shape (batch_size)
        
        Note: Batch is reverse sorted according to text lengths.
        """
        # First group fields across batches
        target_batch, distr1_batch, distr2_batch, messages_batch = [], [], [], []
        for d in data:
            target_batch.append(d['target'])
            distr1_batch.append(d['distr1'])
            distr2_batch.append(d['distr2'])
            messages_batch.append(d['message'])
            
        # Next numericalize the batch and reverse sort (descending order) by
        # text lengths
        messages_batch_numericalized, messages_batch_length = vocab_field.process(messages_batch)
        data = list(zip(
            target_batch, distr1_batch, distr2_batch, messages_batch_numericalized, messages_batch_length
        ))
        data.sort(key=lambda x: x[4], reverse=True)
        target, distr1, distr2, messages, messages_lengths = zip(*data)

        # Stack messages
        messages = torch.stack(messages, 0)
        messages_lengths = torch.stack(messages_lengths, 0)

        # Repeat messages for target, distr1, distr2
        max_msg_size = messages_lengths[0]
        messages = torch.cat([messages, messages, messages], dim=1)
        messages = messages.view(-1, max_msg_size)
        messages_lengths = messages_lengths.unsqueeze(dim=1)
        messages_lengths = torch.cat([messages_lengths, messages_lengths, messages_lengths], dim=1)
        messages_lengths = messages_lengths.view(-1, 1)
        messages_lengths = messages_lengths.squeeze()

        # Cocatenate Images
        target = torch.stack(target, 0).view(-1, 224 * 224 * 3) # flatten
        distr1 = torch.stack(distr1, 0).view(-1, 224 * 224 * 3) # flatten
        distr2 = torch.stack(distr2, 0).view(-1, 224 * 224 * 3) # flatten
        stims = torch.cat([target, distr1, distr2], dim=1)
        stims = stims.view(-1, 3, 224, 224,)

        # Labels
        labels = torch.zeros(len(target_batch), dtype=torch.long)
        return stims, (messages, messages_lengths), labels

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
