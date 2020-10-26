#!/usr/bin/env python
# coding: utf-8

# # Joining all processed data
# This notebook joins all processed data and then saves it in a file for subsequent modeling.

# In[87]:


# Last amended: 24th October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Joining all processed datasets
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[88]:


# 1.0 Libraries
#     (Some of these may not be needed here.)
get_ipython().run_line_magic('reset', '-f')
import numpy as np
import pandas as pd
import gc

# 1.1 Reduce read data size
#     There is a file reducing.py
#      in this folder. A class
#       in it is used to reduce
#        dataframe size
#     (Code modified by me to
#      exclude 'category' dtype)
import reducing

# 1.2 Misc
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[89]:


# 1.3
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[90]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[91]:


# 2.0 Prepare to read data
pathToData = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToData)


# In[92]:


# 2.1 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# In[93]:


# 3.0 Read previous application data first
df = pd.read_csv(
                   'processed_df.csv.zip',
                   nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

df = reducing.Reducer().reduce(df)


# In[94]:


# 3.1
df.shape    # (356251, 262)
df.head(2)


# In[95]:


# 3.2
df.columns
df.drop(columns = ['Unnamed: 0', 'index'], inplace = True)
df.columns


# In[96]:


# 3.3
df.head(2)


# In[97]:


# 3.4 Set SK_ID_CURR as Index
df = df.set_index('SK_ID_CURR')
df.head(2)
df.shape    # (356251, 259)


# In[98]:


# 4.0  Read bureau_agg
bureau_agg = pd.read_csv(
                   'processed_bureau_agg.csv.zip',
                   nrows = num_rows
                   )

# 4.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

bureau_agg = reducing.Reducer().reduce(bureau_agg)


# In[99]:


# 4.1 Set index 
bureau_agg.head(2)
bureau_agg = bureau_agg.set_index("SK_ID_CURR")
bureau_agg.head(2)
bureau_agg.shape    # (305811, 116)


# In[100]:


# 5.0 Join bureau_agg with df
df = df.join(
             bureau_agg,
             how='left',
             on='SK_ID_CURR'
            )


# In[101]:


# 5.1
df.shape    # (356251, 375)
df.head(2)


# In[102]:


# 5.2 Read previous application data
prev_agg = pd.read_csv(
                   'processed_prev_agg.csv.zip',
                   nrows = num_rows
                   )

# 5.3 Reduce memory usage by appropriately
#       changing data-types per feature:

prev_agg = reducing.Reducer().reduce(prev_agg)


# In[103]:


# 5.3 Set Index
prev_agg.shape    # (338857, 250)
prev_agg.head(2)
prev_agg = prev_agg.set_index("SK_ID_CURR")
prev_agg.head(2)
prev_agg.shape    # (338857, 250)


# In[104]:


# 6.0 Join prev_agg with df
df = df.join(prev_agg, how='left', on='SK_ID_CURR')
df.shape    # (356251, 624)
df.head(2)


# In[105]:


# 7.0 Read processed POS data
pos_agg = pd.read_csv(
                   'processed_pos_agg.csv.zip',
                   nrows = num_rows
                   )

# 7.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

pos_agg = reducing.Reducer().reduce(pos_agg)


# In[106]:


# 7.1
pos_agg.shape    # (337252, 19)
pos_agg.head(2)
pos_agg = pos_agg.set_index("SK_ID_CURR")
pos_agg.head(2)
pos_agg.shape   # (337252, 18)


# In[107]:


# 7.2 Join POS with df
df = df.join(
             pos_agg,
             how='left',
             on='SK_ID_CURR'
             )

df.shape    # (356251, 642)
df.head(2)


# In[108]:


# 8.0 Read processed installments data
ins_agg = pd.read_csv(
                   'processed_ins_agg.csv.zip',
                   nrows = num_rows
                   )

# 8.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

ins_agg = reducing.Reducer().reduce(ins_agg)


# In[109]:


# 8.1 Set index
ins_agg.shape    # (339587, 26)
ins_agg.head(2)
ins_agg = ins_agg.set_index("SK_ID_CURR")
ins_agg.head(2)
ins_agg.shape   # (339587, 25)


# In[110]:


# 9.0 Join Installments data with df
df = df.join(ins_agg, how='left', on='SK_ID_CURR')
df.shape    # (356251, 667)
df.head(2)


# In[111]:


# 10.0 Read Credit card data
cc_agg = pd.read_csv(
                   'processed_creditCard_agg.csv.zip',
                   nrows = num_rows
                   )

# 10.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

cc_agg = reducing.Reducer().reduce(cc_agg)


# In[112]:


# 10.1 Set Index
cc_agg.shape    # (103558, 142)
cc_agg.head(2)
cc_agg = cc_agg.set_index("SK_ID_CURR")
cc_agg.head(2)
cc_agg.shape   # (103558, 141)


# In[113]:


# 11. Join Credit card data with df
df = df.join(cc_agg, how='left', on='SK_ID_CURR')
df.shape    # (356251, 808)
df.head(2)


# In[114]:


# 11.1 Save the results for subsequent use:
df.to_csv("processed_df_joined.csv.zip", compression = "zip")   


# In[ ]:


##################

