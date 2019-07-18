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

#%%
'''Create the BERT classification model'''

  """build classification model."""

def build_model(predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):

	'''model architecture config'''

	bert_module = hub.Module(
      	BERT_MODEL_HUB,
      	trainable=True)

  	bert_inputs = dict(
      	input_ids=input_ids,
      	input_mask=input_mask,
      	segment_ids=segment_ids)

  	bert_outputs = bert_module(
    	inputs=bert_inputs,
      	signature="tokens",
      	as_dict=True)


    '''layer config'''

	# Will classify entire sentence over all labels
  	output_layer = bert_outputs["pooled_output"]
  	hidden_size = output_layer.shape[-1].value

    # initialise layer weights & bias
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
         initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())	 


   '''training/inference config'''

    with tf.variable_scope("loss"):

        # Add layer dropout of 0.1 per layer 
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        
        # when prediction will output labels and proabilities 
        if predicting:
            # can change depending on desired feedback
            return (predicted_labels, log_probs) 

        # when trainng modle willcompute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, predicted_labels, log_probs)
     

#%%
 '''Create the training and prediction functions'''

def model_fn_builder(num_labels, learning_rate, num_train_steps,num_warmup_steps):
  
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

    '''TODO - evalution function'''

    # return the function
    return model_fn    

#%%
'''Set training hyperpramameters'''

LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1

# Model checkpoints
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100         

'''TODO - save complete trained BERT model'''


#%%
'''Define number of step in training process'''

num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

#%%
'''Confirm Output Directory & No Checkpoints'''

run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)


#%%
'''Define the Model Object'''

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

#%%
#%%
'''Define the Training Input Function'''

train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True)    