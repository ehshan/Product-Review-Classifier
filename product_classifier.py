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

train_out = 'train_out'
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
'''Assigned input labels for BERT data'''

DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'star_rating'
label_list = [1, 2, 3, 4, 5]


#%%
'''Transform data into BERT readable objects'''

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)



#%%
'''Transform data to match BERT pre-trained data '''

# Load the pre-trained uncased model from tensorflow hub

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()


#%%
'''Convert features to BERT understandable'''

# set max sequence length
MAX_SEQ_LENGTH = 100

train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
