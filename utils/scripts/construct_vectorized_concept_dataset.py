"""
File: construct_vectorized_concept_dataset.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 8, 2019
Description: Create dataset with samples grouped into individual concepts.
"""
import pandas as pd

DATA_DIR = "./data/concept/{}/vectorized/"
ORIG_TSV_NAME = "concat_informative_dataset.tsv"


def main():
    """
    Creates all concept tsv files.
    """
    create_tsv(DATA_DIR.format("train"), ORIG_TSV_NAME)
    create_tsv(DATA_DIR.format("test"), ORIG_TSV_NAME)
    create_tsv(DATA_DIR.format("val"), ORIG_TSV_NAME)

def create_tsv(data_dir, orig_tsv_name):
    """
    Creates an individual concept df from the df in orig_tsv_name and writes it
    to data_dir.
    """
    concept_df = create_concept_df(data_dir + orig_tsv_name)
    concept_df.to_csv(data_dir + "concept_dataset", sep="\t")

def create_concept_df(orig_tsv):
    df_orig = pd.read_csv(orig_tsv, sep="\t")
    return df_orig

if __name__ == '__main__':
    main()
