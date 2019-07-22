"""
File: construct_vectorized_concept_dataset.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 8, 2019
Description: Create dataset with samples grouped into individual concepts.
"""
import pandas as pd

DATA_DIR = "./data/{}/{{}}/vectorized/"
ORIG_TSV_NAME_CONCEPT = "concat_informative_dataset.tsv"
ORIG_TSV_NAME_REF = "dataset.tsv"
NEW_TSV_NAME_CONCEPT = "concept_dataset.tsv"
NEW_TSV_NAME_REF = "ref_dataset.tsv"
NEW_TSV_NAME_CONCEPT_UNIQUE = "unique_concept_dataset.tsv"


def main_concept():
    """
    Creates all concept tsv files.
    """
    data_dir = DATA_DIR.format("concept")
    create_tsv(data_dir.format("train"), ORIG_TSV_NAME_CONCEPT, ref=False)
    create_tsv(data_dir.format("test"), ORIG_TSV_NAME_CONCEPT, ref=False)
    create_tsv(data_dir.format("val"), ORIG_TSV_NAME_CONCEPT, ref=False)

def main_ref():
    """
    Creates all ref tsv files.
    """
    data_dir = DATA_DIR.format("reference/pilot_coll1")
    create_tsv(data_dir.format("train"), ORIG_TSV_NAME_REF, ref=True)
    create_tsv(data_dir.format("test"), ORIG_TSV_NAME_REF, ref=True)
    create_tsv(data_dir.format("val"), ORIG_TSV_NAME_REF, ref=True)


def create_tsv(data_dir, orig_tsv_name, ref=False, unique_concepts=True):
    """
    Creates a dataset of concepts or reference games such that each stimulus rep
    is its own column
    If unique_concepts is true, then the concept dataset only has one description per concept
    """
    if ref:
        df = create_ref_df(data_dir + orig_tsv_name)
        new_tsv_name = NEW_TSV_NAME_REF
    else:
        if unique_concepts:
            df = create_unique_concept_df(data_dir + orig_tsv_name)
            new_tsv_name = NEW_TSV_NAME_CONCEPT_UNIQUE
        else:
            df = create_concept_df(data_dir + orig_tsv_name)
            new_tsv_name = NEW_TSV_NAME_CONCEPT
    df.to_csv(data_dir + new_tsv_name, sep='\t', index=False)

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

def create_unique_concept_df(orig_tsv):
    '''
    Creates a dataset of concepts where no concept is repeated
    (In the original dataset each concept is described ~10 times)
    '''
    pivot = create_concept_df(orig_tsv)
    stim_cols = [col for col in pivot if type(col)==int]
    unique_concepts_pivot = pivot.drop_duplicates(subset=stim_cols, keep='first')
    return unique_concepts_pivot

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

def create_ref_df(orig_tsv):
    '''
    Creates a reference game dataset of an identical format to the concept dataset
    created above: each stimulus is a single column, in addition to one column for the text,
    and one column for the labels (which in this case are all precisely 001)
    '''
    df = pd.read_csv(orig_tsv, sep='\t')
    distr1_cols = [col for col in df if 'distr1' in col]
    df['distr1'] = df[distr1_cols].apply(lambda x: ''.join(x.astype(str)),
                                         axis=1)
    distr2_cols = [col for col in df if 'distr2' in col]
    df['distr2'] = df[distr2_cols].apply(lambda x: ''.join(x.astype(str)),
                                         axis=1)
    target_cols = [col for col in df if 'target' in col]
    df['target'] = df[target_cols].apply(lambda x: ''.join(x.astype(str)),
                                         axis=1)
    df['labels'] = '001'
    # XXX Remove weird tildes
    df['message'] = df['message'].apply(lambda x: x.replace('~', ''))
    # Remove extraneous stimulus columns and reorder
    df = df[['message', 'labels', 'distr1', 'distr2', 'target']]
    # Rename to match concept dataset
    df = df.rename(columns=dict(message='text', distr1=0, distr2=1, target=2))
    return df

if __name__ == '__main__':
    # main_concept()
    main_ref()
