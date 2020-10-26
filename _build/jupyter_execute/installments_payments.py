#!/usr/bin/env python
# coding: utf-8

# # Install payments
# ## About data
# <blockquote>It is payment history for previous loans at <i>Home Credit</i>. There is one row for every made payment and one row for every missed payment.</blockquote> 

# ## Feature explanations
# <blockquote><p style="font-size:13px">
# SK_ID_PREV : 	ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)<br>
# SK_ID_CURR: 	ID of loan in our sample<br>
# NUM_INSTALMENT_VERSION: 	Version of installment calendar (0 is for credit card) of previous credit. Change of installment version from month to month signifies that some parameter of payment calendar has changed<br>
# NUM_INSTALMENT_NUMBER: 	On which installment we observe payment<br>
# DAYS_INSTALMENT: 	When the installment of previous credit was supposed to be paid (relative to application date of current loan)<br>
# DAYS_ENTRY_PAYMENT: 	When was the installments of previous credit paid actually (relative to application date of current loan)<br>
# AMT_INSTALMENT: 	What was the prescribed installment amount of previous credit on this installment<br>
# AMT_PAYMENT: 	What the client actually paid on previous credit on this installment</p></blockquote>
# 

# In[47]:


# Last amended: 24th October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing installment_payments dataset
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[48]:


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


# In[49]:


# 1.3
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[50]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[51]:


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


# In[52]:


# 3.0 Prepare to read data
pathToData = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToData)


# In[53]:


# 2.2 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# In[56]:


# 3.0 Read previous application data first
ins = pd.read_csv(
                   'installments_payments.csv.zip',
                   nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

ins = reducing.Reducer().reduce(ins)


# In[57]:


# 3.1
ins.shape   # (13605401, 8)
ins.head()


# In[58]:


# 3.2 No object type column
ins.dtypes.value_counts()


# In[59]:


# 3.3 OHE any object column
ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)


# In[60]:


# 3.3.1 This dataset does not have any object feature
cat_cols


# In[61]:


# 3.4
ins.shape   # 13605401, 8)
ins.head()


# In[62]:


# 4.0 Percentage and difference paid in each installment (amount paid and installment value)
ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']


# In[63]:


# 4.1 Days past due and days before due (no negative values)
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)


# In[64]:


# 4.2 How to perform aggregations?
#     For numeric columns
aggregations = {
                 'NUM_INSTALMENT_VERSION': ['nunique'],
                 'DPD': ['max', 'mean', 'sum'],
                 'DBD': ['max', 'mean', 'sum'],
                 'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
                 'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
                 'AMT_INSTALMENT': ['max', 'mean', 'sum'],
                 'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
                 'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
               }

# 4.2.1 For categorical columns
for cat in cat_cols:
    aggregations[cat] = ['mean']  


# In[65]:


# 4.2.2
aggregations


# In[66]:


# 4.3 Perform aggregation now
grouped = ins.groupby('SK_ID_CURR')
ins_agg= grouped.agg(aggregations)


# In[67]:


# 4.4
ins_agg.shape
ins_agg.head()


# In[68]:


# 4.5 Rename columns
ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])


# In[69]:


# 4.6
ins_agg.shape
ins_agg.head()


# In[70]:


# 4.7 Create one more column. Per client how many installments accounts
ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()


# In[72]:


# 4.8
ins_agg.shape
ins_agg.head()


# In[73]:


# 5.0 Save the results for subsequent use:
ins_agg.to_csv("processed_ins_agg.csv.zip", compression = "zip")   


# In[ ]:


##############

