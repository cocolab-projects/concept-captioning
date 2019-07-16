"""
File: construct_vectorized_concept_dataset.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 8, 2019
Description: Create dataset with samples grouped into individual concepts.
"""
import pandas as pd

DATA_DIR = "./data/concept/{}/vectorized/"
ORIG_TSV_NAME = "concat_informative_dataset.tsv"
NEW_TSV_NAME = "concept_dataset.tsv"


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
    concept_df.to_csv(data_dir + NEW_TSV_NAME, sep="\t", index=False)

def create_concept_df(orig_tsv):
    df = pd.read_csv(orig_tsv, sep="\t")
    # Inspired by https://stackoverflow.com/questions/33098383/merge-multiple-column-values-into-one-column-in-python-pandas
    # Note that only feature columns have '-' in the header
    df['concat_rep'] = df[[col for col in df if "-" in col]].apply(
        lambda feature: ''.join(feature.astype(str)),
        axis = 1)
    # Numericalize teacher labels
    df['true_label'] = df['true_label'].apply(lambda x: str(int(x)))
    # Make binary strings of all labels for each concept
    concat_labels_ground = df.groupby('text')['true_label'].apply(lambda x: ''.join(x))
    # Pivot each row to be a concept
    pivot = pd.pivot_table(df, values = 'concat_rep', index=['text'], columns = ['stim_num'], aggfunc = 'first')
    # Add back in the text field (since it was used as the index for pivoting)
    pivot['text'] = pivot.index
    # Add strings of labels to pivoted concept table
    pivot['labels'] = concat_labels_ground
    # Move special fields to front just for aesthetics
    cols = pivot.columns.tolist()
    pivot = pivot[cols[-2:] + cols[:-2]]
    return pivot

def create_concept_df_all_labels(orig_tsv):
    df = pd.read_csv(orig_tsv, sep="\t")
    # Inspired by https://stackoverflow.com/questions/33098383/merge-multiple-column-values-into-one-column-in-python-pandas
    # Note that only feature columns have '-' in the header
    df['concat_rep'] = df[[col for col in df if "-" in col]].apply(
        lambda feature: ''.join(feature.astype(str)),
        axis = 1)
    # Numericalize teacher labels
    df[['teacher_label', 'student_label', 'true_label']] = df[['teacher_label', 'student_label', 'true_label']].apply(lambda x: x.astype(int).astype(str))
    #df['student_label'] = df['student_label'].apply(lambda x: str(int(x)))
    #df['teacher_label'] = df['teacher_label'].apply(lambda x: str(int(x)))
    # Make binary strings of all labels for each concept
    concat_labels_teacher = df.groupby('text')['teacher_label'].apply(lambda x: ''.join(x))
    concat_labels_student = df.groupby('text')['student_label'].apply(lambda x: ''.join(x))
    concat_labels_ground = df.groupby('text')['true_label'].apply(lambda x: ''.join(x))
    # Pivot each row to be a concept
    pivot = pd.pivot_table(df, values = 'concat_rep', index=['text'], columns = ['stim_num'], aggfunc = 'first')
    # Add back in the text field (since it was used as the index for pivoting)
    pivot['text'] = pivot.index
    # Add strings of labels to pivoted concept table
    pivot['teacher_labels'] = concat_labels_teacher
    pivot['student_labels'] = concat_labels_student
    pivot['true_labels'] = concat_labels_ground
    # Convert column label types to strings (feature vec labels were ints)
    pivot.columns = pivot.columns.astype(str)
    pivot['all_labels'] = pivot[[col for col in pivot if "labels" in col]].apply(
        lambda label_list: '|'.join(label_list),
        axis = 1)
    # Move special fields to front just for aesthetics
    cols = pivot.columns.tolist()
    pivot = pivot[cols[-5:] + cols[:-5]]
    return pivot


if __name__ == '__main__':
    main()
