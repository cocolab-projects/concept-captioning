'''
Author: Will Schwarzer
Date: August 13, 2019
Contains utility functions for working with torch tensors
'''

import torch

def to_onehot(y, n=5):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot
