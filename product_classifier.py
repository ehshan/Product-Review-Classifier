#%%[markdown]

## BERT Classifier for Product Reviews

#%%
#%%
'''Import Dependencies'''

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

'''TODO: Docker machine learning enviroment'''

#%%
'''Import BERT'''

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

#%%
'''Download and Load Data Function'''

from tensorflow import keras
import os
import re


#%%
'''Set output directory for model and eval files '''

train_out = 'rain_out'
OUTPUT_DIR = train_out

#%%

'''Load data from remote into dataframe '''

def load_data_from_remote(force_download = False):
    
    # load data from url
    dataset = tf.keras.utils.get_file(
        fname = "sample_us.tsv", 
        origin = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/sample_us.tsv", 
        extract = False)
    
    # relevant fields from data
    fields = ['review_id', 'star_rating', 'review_body']
    
    # read data to df
    df = pd.read_csv(dataset, sep='\t', header=0, skipinitialspace=True, usecols=fields)

    return df

#%%
'''Spit data to train/test sets '''

df = load_data_from_remote()
train, test = train_test_split(df, test_size=0.2)


#%%
