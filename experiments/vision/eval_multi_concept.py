 #!/usr/bin/env python 
"""
File: eval_concept.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: May 28, 2019
Description: Evaluate a pre-trained reference model.
"""
import uuid
import json
import matplotlib
import os
import pandas as pd
import numpy as np
import sys
import time

from experiments.utils import AccuracyMeter, set_seeds
from experiments.utils import load_multi_task_student_checkpoint as load_student_checkpoint
from utils.dataloaders.vision.concept_load_dataset import get_data_loader

import torch
import torchtext.data as data
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='model name')
    parser.add_argument('--split', type=str, default='val',
                        help='split to evaluate [default: val]')    
    parser.add_argument('--model-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/multi_task/models/',
                        help='where to save models [default: /mnt/fs5/schopra/ratchet/lfl/student/multi_task/models]')
    parser.add_argument('--data-dir', type=str, default='./data/concept/{}/vision/concat_informative_dataset.tsv',
                        help='file template for dataset')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 32]')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # Set seeds
    set_seeds()

    # Load Data
    data_loader, vocab = get_data_loader('./data/concept/', args.split, batch_size=args.batch_size)

    # Evaluation
    def eval(student):
        acc_meter = AccuracyMeter()
        pbar = tqdm(total=len(data_loader))

        if args.cuda:
            student = student.to('cuda')
        student.eval()
        results = []       

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images, (texts, text_lengths), student_labels, teacher_labels, true_labels, data_ids = batch
                if args.cuda:
                    images = images.to('cuda')
                    texts = texts.to('cuda')
                    text_lengths = text_lengths.to('cuda')
                    true_labels = true_labels.to('cuda')
                    teacher_labels = teacher_labels.to('cuda')
                    student_labels = student_labels.to('cuda')
                all_labels = {
                    'ground_truth': true_labels,
                    'student': student_labels,
                    'teacher': teacher_labels
                }
                logits, _ = student(texts, images, text_lengths, use_concept=True)
                _, y_hat  = torch.max(logits, dim=1)
                r = pd.DataFrame(data_ids, columns=['gameid', 'rule_idx'])
                r['y_hat'] = y_hat.tolist()
                r['teacher'] = teacher_labels.tolist()
                r['student'] = student_labels.tolist()
                r['ground_truth'] = true_labels.tolist() 
                results.append(r)
                acc_meter.update(logits, all_labels, cuda=args.cuda, vision=True)
                pbar.update()
        pbar.close()
        print('{}  Accuracies:'.format(args.split))
        acc_meter.print()
        results = pd.concat(results)
        return acc_meter, results

    # Model Training
    print("Loading best model from disk ...")
    student = load_student_checkpoint(os.path.join(args.model_dir, args.name, 'model_weights', 'concept_model_best.pth.tar'), use_cuda=args.cuda)
    acc_meter, results = eval(student)
    results.to_csv('~/results_{}.tsv'.format(args.split), sep='\t', index=False)
