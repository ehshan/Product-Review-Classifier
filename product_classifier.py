#%%[markdown]

## BERT Classifier for Product Reviews

#%%[markdown]

### Create Enviroment

'''TODO: Docker machine learning enviroment'''

#%%
'''Import Dependencies'''

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime


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


#%%[markdown]

### Download, Clean & Process Data

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
    """Get the vocab file and casing info from the Hub module"""
      
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


#%%[markdown]

### Create the Model


#%%
'''Create the BERT classification model'''

def build_model(predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):

    """Model Architecture Config"""

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


    '''Layer Config'''

    # Will classify sentence over all labels
    output_layer = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1].value

    # initialise layer weights and bias
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    
    '''Output Config'''

    with tf.variable_scope("loss"):

        # Add layer dropout to 0.1 per layer
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        
        # when predicting model will output label and probability
        if predicting:
            return (predicted_labels, log_probs)

        # When training/eval model will output loss
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)



#%%
 '''Create the training and prediction functions'''

def model_fn_builder(num_labels, learning_rate, num_train_steps,num_warmup_steps):
  
    """Model function for training, evaluation and predictions"""

    def model_fn(features, labels, mode, params): 

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        
        '''prediction function'''

        predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        '''training and evalution function'''
        
        if not predicting:

            (loss, predicted_labels, log_probs) = build_model(
                predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)


            '''define model eval metrics'''

            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels) 
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)   
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)  
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)
            

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = build_model(
                predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
                }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # return the function
    return model_fn    

#%%[markdown]

### Set the Training Enviroment

#%%
'''Set training hyperpramameters'''
BATCH_SIZE = 32
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
'''Define the Training Input Function'''

train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)  

#%%
'''Train Model'''

print(f'Trainig Classifier')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('{}{}'.format("Training took time ", datetime.now() - current_time))    
