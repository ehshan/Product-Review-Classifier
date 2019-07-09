#%%[markdown]

## BERT Classifier for Product Reviews

#%%
#%%
'''Import Dependencies'''
from sklearn.model_selection import train_test_split
import pandas as pd
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
