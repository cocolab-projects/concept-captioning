"""
File: split_vision_dataset.py
Author: Sahil Chopra (schopra8@stanford.edu)
Date: April 29, 2019
Description: Split vision dataset according to pre-determine allocations.
"""

import csv
import glob
import pandas as pd
import os
from shutil import copyfile

def split_concept_imgs(vision_dir, output_dir, rules_json):
    """ Copy images form vision_dir to the output_dir
        if they belong to rules listed in rules_json.
    """
    vision_ids_dir = os.path.join(vision_dir, 'ids')
    vision_imgs_dir = os.path.join(vision_dir, 'png_imgs')

    # Make ouptut directories
    output_ids_dir = os.path.join(output_dir, 'ids')
    output_imgs_dir = os.path.join(output_dir, 'imgs')
    if not os.path.exists(output_ids_dir):
        os.makedirs(output_ids_dir)
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)

    rule_names = pd.read_json(rules_json).transpose()['name'].tolist()

    id_files = glob.glob(os.path.join(vision_ids_dir, "*.csv"))
    for id_file in id_files:
        rule_name = os.path.splitext(os.path.split(id_file)[1])[0]
        if rule_name in rule_names:
            print ("Copying data for: {}".format(rule_name))
            # copy id file
            dst = os.path.join(output_dir, 'ids', os.path.basename(id_file))
            copyfile(id_file, dst)

            # copy image files
            img_ids = pd.read_csv(id_file)['id'].tolist()
            for img_id in img_ids:
                img_fn = '{}.png'.format(os.path.splitext(os.path.split(img_id)[1])[0])
                img_src = os.path.join(vision_imgs_dir, img_fn)
                img_dst = os.path.join(output_imgs_dir, img_fn)
                copyfile(img_src, img_dst)


def split_ref_imgs(vision_dir, output_dir, dataset_file):
    """ Copy images form vision_dir if they are listed in the dataset.
    """
    vision_imgs_dir = os.path.join(vision_dir, 'png_imgs')

    # Make output directories
    output_imgs_dir = os.path.join(output_dir, 'imgs')
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    print("Copying images into: {}".format(output_imgs_dir))


    # Read Dataset
    dataset = pd.read_csv(dataset_file, sep='\t')
    
    def copy_files(images_list):
        for img_fn in images_list:
            img_src = os.path.join(vision_imgs_dir, img_fn)
            img_dst = os.path.join(output_imgs_dir, img_fn)
            copyfile(img_src, img_dst)
    copy_files(dataset['distr1'].tolist())
    copy_files(dataset['distr2'].tolist())
    copy_files(dataset['target'].tolist())




if __name__ == '__main__':
    # # Train
    # split_concept_imgs(
    #     vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/raw/stims/test_stim/vision',
    #     output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/train/vision',
    #     rules_json='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/train/rules.json'  
    # )
    # # Validation
    # split_concept_imgs(
    #     vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/raw/stims/test_stim/vision',
    #     output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/val/vision',
    #     rules_json='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/val/rules.json'  
    # )
    # # Test
    # split_concept_imgs(
    #     vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/raw/stims/test_stim/vision',
    #     output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/test/vision',
    #     rules_json='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/concept/test/rules.json'  
    # )

    # Train
    split_ref_imgs(
        vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/raw/vision',
        output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/train/vision',
        dataset_file='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/train/vision/dataset.tsv',  
    )
    # Validation
    split_ref_imgs(
        vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/raw/vision',
        output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/val/vision',
        dataset_file='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/val/vision/dataset.tsv',
    )
    # Test
    split_ref_imgs(
        vision_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/raw/vision',
        output_dir='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/test/vision',
        dataset_file='/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/data/reference/pilot_coll1/test/vision/dataset.tsv',  
    )