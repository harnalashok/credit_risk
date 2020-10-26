#!/usr/bin/env python
# coding: utf-8

# # POS Cash balance
# ## About the data
# <blockquote>POS_CASH_BALANCE: Monthly data about previous point of sale or cash loans clients have had with <u>Home Credit</u>. Each row is <i>one month</i> of a previous point of sale or cash loan, and a single previous loan can have many rows. This dataset contrasts with <i>bureau_balance</i> dataset where monthly installments were of loans with <u>bureau</u>.</blockquote>

# ## Feature explanations
# <blockquote><p style="font-size:13px"> 
# SK_ID_PREV : 	ID of previous credit in Home Credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)	
# SK_ID_CURR: 	ID of loan in our sample	
# MONTHS_BALANCE: 	Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly )	
# CNT_INSTALMENT: 	Term of previous credit (can change over time)	
# CNT_INSTALMENT_FUTURE: 	Installments left to pay on the previous credit	
# NAME_CONTRACT_STATUS: 	Contract status during the month	
# SK_DPD: 	DPD (days past due) during the month of previous credit	
# SK_DPD_DEF: 	DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit</p></blockquote>

# In[ ]:


# Last amended: 24th October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing POS_CASH_balance dataset
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[22]:


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


# In[24]:


# 1.3
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[25]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[26]:


# 2.0 Onehot encoding (OHE) function. Uses pd.get_dummies()
#     i) To transform 'object' columns to dummies. 
#    ii) Treat NaN as one of the categories
#   iii) Returns transformed-data and new-columns created

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df,
                        columns= categorical_columns,
                        dummy_na= nan_as_category       # Treat NaNs as category
                       )
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[27]:


# 3.0 Prepare to read data
pathToData = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToData)


# In[28]:


# 3.1 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# ## About the data
# <blockquote>POS_CASH_BALANCE: Monthly data about previous point of sale or cash loans clients have had with <u>Home Credit</u>. Each row is <i>one month</i> of a previous point of sale or cash loan, and a single previous loan can have many rows. This dataset contrasts with <i>bureau_balance</i> dataset where monthly installments were of loans with <u>bureau</u>.</blockquote>

# In[29]:


# 3.2 Read previous application data first
pos = pd.read_csv(
                   'POS_CASH_balance.csv.zip',
                   nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

pos = reducing.Reducer().reduce(pos)


# In[30]:


# 3.3
pos.shape    # (rows: 1,00,01358, cols: 8)
pos.head()


# ## Feature explanations
# SK_ID_PREV : 	ID of previous credit in Home Credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)	
# SK_ID_CURR: 	ID of loan in our sample	
# MONTHS_BALANCE: 	Month of balance relative to application date (-1 means the information to the freshest monthly snapshot, 0 means the information at application - often it will be the same as -1 as many banks are not updating the information to Credit Bureau regularly )	
# CNT_INSTALMENT: 	Term of previous credit (can change over time)	
# CNT_INSTALMENT_FUTURE: 	Installments left to pay on the previous credit	
# NAME_CONTRACT_STATUS: 	Contract status during the month	
# SK_DPD: 	DPD (days past due) during the month of previous credit	
# SK_DPD_DEF: 	DPD during the month with tolerance (debts with low loan amounts are ignored) of the previous credit	
# 

# In[31]:


# 3.3.1 There is one object type
pos.dtypes.value_counts()


# In[32]:


# 4.0 Transform object type columns to OHE
pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)


# In[33]:


# 4.1
pos.shape    # (10001358, 17)
pos.head()


# In[34]:


# 4.2 New columns are:
cat_cols


# In[35]:


# 4.3 How to aggregate features:
#     Note CNT_INSTALMENT and CNT_INSTALMENT_FUTURE
#       do not find place:

aggregations = {
                'MONTHS_BALANCE': ['max', 'mean', 'size'],
                'SK_DPD':         ['max', 'mean'],
                'SK_DPD_DEF':     ['max', 'mean']
               }

# 4.3.1
for cat in cat_cols:
    aggregations[cat] = ['mean']
    


# In[36]:


# 4.3.2 Full dictionary
aggregations


# In[37]:


# 5.0 Aggregate now
grouped = pos.groupby('SK_ID_CURR')
pos_agg = grouped.agg(aggregations)


# In[38]:


# 5.1
pos_agg.shape     # (337252, 17)
pos_agg.head()
pos_agg.columns


# In[39]:


# 5.2 Rename multiindex columns
pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])


# In[40]:


# 5.3
pos_agg.columns
pos_agg.head()


# In[41]:


# 5.4 Count pos cash accounts
#     Per client how many entries/rows exist
pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()


# In[42]:


pos_agg.head()


# In[43]:


# 6.0 Save the results for subsequent use:
pos_agg.to_csv("processed_pos_agg.csv.zip", compression = "zip")   
    


# In[ ]:


######################

