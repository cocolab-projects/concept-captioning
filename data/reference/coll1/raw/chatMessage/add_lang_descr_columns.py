"""
add_lang_descr_columns.py
Add "messageType" and "uttClass" to files so that they can be 
properly parsed downstream
"""
import glob
import json
import os
import numpy as np
import pandas as pd

msgFiles = glob.glob(os.path.join('.', "*.tsv"))
for f in msgFiles:
    msgs = pd.read_csv(f, sep='\t')
    msgs['messageType'] = ''
    msgs['uttClass'] = '' 
    msgs.to_csv(f, sep='\t', index=False)