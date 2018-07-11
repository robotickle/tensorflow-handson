
# coding: utf-8

# # Trying out features

# **Learning Objectives:**
#   * Improve the accuracy of a model by adding new features with the appropriate representation

# The data is based on 1990 census data from California. This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.

# ## Set Up
# In this first cell, we'll load the necessary libraries.

# In[1]:


import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# Next, we'll load our data set.

# In[2]:


df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")


# ## Examine and split the data
# 
# It's a good idea to get to know your data a little bit before you work with it.
# 
# We'll print out a quick summary of a few useful statistics on each column.
# 
# This will include things like mean, standard deviation, max, min, and various quantiles.

# In[3]:


df.head()


# In[4]:


df.describe()


# Now, split the data into two parts -- training and evaluation.

# In[11]:


np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]


# In[13]:


len(traindf)


# In[14]:


len(evaldf)


# ## Training and Evaluation
# 
# In this exercise, we'll be trying to predict **median_house_value** It will be our label (sometimes also called a target).
# 
# We'll modify the feature_cols and input function to represent the features you want to use.
# 
# Hint: Some of the features in the dataframe aren't directly correlated with median_house_value (e.g. total_rooms) but can you think of a column to divide it by that we would expect to be correlated with median_house_value?

# In[18]:


def add_more_features(df):
  # TODO: Add more features to the dataframe we can add total_rooms, total_bathrooms, basement,
  df['num_rooms']=df['total_rooms']/df['households']
  df['num_bedrooms']=df['total_bedrooms']/df['households']
  tRooms = df['num_rooms']
  tBedrooms = df['num_bedrooms']
  len(tRooms)
  len(tBedrooms)
  return df


# In[19]:


# Create pandas input function
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = add_more_features(df),
    y = df['median_house_value'] / 100000, # will talk about why later in the course
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )


# In[35]:


# Define your feature columns
def create_feature_cols():
  return [
    tf.feature_column.numeric_column('housing_median_age'),
    # TODO: Define additional feature columns
    # Hint: Are there any features that would benefit from bucketizing?
    # 1 degree is esentially around 100KM, which is good.
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), boundaries = np.arange(32,42,1).tolist()),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'), boundaries = np.arange(-124,114,1).tolist()),
    tf.feature_column.numeric_column('num_rooms'),
    tf.feature_column.numeric_column('num_bedrooms'),
    tf.feature_column.numeric_column('median_income')
    
  ]


# In[42]:


# Create estimator train and evaluate function
def train_and_evaluate(output_dir, num_train_steps):
  # TODO: Create tf.estimator.LinearRegressor, train_spec, eval_spec, and train_and_evaluate using your feature columns
  estimator = tf.estimator.LinearRegressor(model_dir = output_dir, feature_columns = create_feature_cols())
  train_spec = tf.estimator.TrainSpec(
                       input_fn = make_input_fn(traindf,8),
                       max_steps = num_train_steps)
  eval_spec = tf.estimator.EvalSpec(
                       input_fn = make_input_fn(evaldf,8),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  


# In[43]:


# Launch tensorboard
from google.datalab.ml import TensorBoard

OUTDIR = './trained_model'
TensorBoard().start(OUTDIR)


# In[44]:


# Run the model
shutil.rmtree(OUTDIR, ignore_errors = True)
train_and_evaluate(OUTDIR, 2000)


# ## Inspect Weights
# 
# Because this is a simple linear model, we can determine feature importance by printing and inspecting the learned weights in our checkpoint file.
# 
# Ignore weights with 'Ftrl' in the tensor name

# In[ ]:


from tensorflow.python.tools import inspect_checkpoint
inspect_checkpoint.print_tensors_in_checkpoint_file("trained_model/model.ckpt-2000", tensor_name='', all_tensors=True)


# Which features were the most predictive? Which are postively correlated? Which are negatively correlated?
# 
# Hint: The magnitude of a feature's weight indicates its importance. The polarity indicates whether it's postively or negatively correlated to the label.
