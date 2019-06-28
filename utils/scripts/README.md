# scripts
Here, we have a mish-mash of 1-off scripts.

## Dataset Construction
Note that `construct_vectorized_dataset.py` was written before `construct_vision_dataset.py`. For now, the latter relies on the former being run, i.e. there must first be a vectorized dataset of train/val/test splits with msgs.tsv and responses.tsv already generated. The latter script just piggy backs on the same splits and formats the same dataset to refer to image files rather than binary feature vectors.