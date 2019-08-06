#%%[markdown]

## Save and Restore Classifer

#### Can be use to predictor or retrain on aditional data

#%%[markdown]

### Create Enviroment

#%%

'''Import Session Variables '''

import dill

# all variables for trained BERT classifer as Jupyter Kernal
dill.load_session('product_classifier.db')

#%%
