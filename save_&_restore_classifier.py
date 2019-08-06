#%%[markdown]

## Save and Restore Classifer

#### Can be use to predictor or retrain on aditional data

#%%[markdown]

### Create Enviroment

#%%
'''Set output directory for model files '''

model_out = 'model_out'
OUTPUT_DIR = model_out


#%%
'''Import Session Variables '''

import dill

# all variables for trained BERT classifer as Jupyter Kernal
dill.load_session('product_classifier.db')


#%%[markdown]

### Used Save model to predict


#%%
'''Define the remote url of sample data '''

product_dir = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/'
sample_file = 'sample_us.tsv'


#%%
'''Load data from remote list '''

def load_sample_from_remote(force_download = False):
    
    # load data from url
    sample_dataset = tf.keras.utils.get_file(
        fname = sample_file, 
        origin = '{}{}'.format(product_dir, sample_file), 
        extract = False)

    fields = ['review_body']
    
        # read data to df
    df = pd.read_csv(sample_dataset, sep='\t', header=0, skipinitialspace=True, usecols=fields, encoding='utf-8')
    
    sample = df.values.tolist()

    # fatten
    sample = [item for items in sample for item in items]

    return sample

    
#%%
'''Assign sample list'''

review_samples = load_sample_from_remote()


#%%
'''Predict class for all sample reviews'''


def predict_class(sentences):
    labels = ['1', '2', '3', '4', '5']
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 4) for x in sentences]
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(sentences, predictions)]



sample_predictions = predict_class(review_samples)

sample_predictions

#%%[markdown]

### Save the trained estimator

#%%
'''Save Estimator '''

'''TODO: Fix issue with feature_spec datatypes'''

def serving_input_receiver_fn():

    feature_spec = {
        "input_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
        "input_mask" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
        "segment_ids" : tf.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
        "label_ids" :  tf.FixedLenFeature([], tf.int64)
    }

    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')
    print(serialized_tf_example.shape)

    receiver_tensors = {'example': serialized_tf_example}
    
    features = tf.parse_example(serialized_tf_example, feature_spec)
    
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


estimator._export_to_tpu = False  

estimator.export_saved_model(export_dir_base = OUTPUT_DIR , serving_input_receiver_fn = serving_input_receiver_fn)
	
