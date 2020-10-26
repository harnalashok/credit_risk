#!/usr/bin/env python
# coding: utf-8

# # Credit Card balance data
# ## About the Dataset
# <blockquote>credit_card_balance: It is monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.</blockquote>

# ## Feature Explanations
# <blockquote><p style="font-size:13px">
# SK_ID_PREV : 	ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loans in Home Credit)<br>		
# SK_ID_CURR: 	ID of loan in our sample<br>		
# MONTHS_BALANCE: 	Month of balance relative to application date (-1 means the freshest balance date)<br>	
# AMT_BALANCE: 	Balance during the month of previous credit<br>		
# AMT_CREDIT_LIMIT_ACTUAL: 	Credit card limit during the month of the previous credit<br>		
# AMT_DRAWINGS_ATM_CURRENT: 	Amount drawing at ATM during the month of the previous credit<br>		
# AMT_DRAWINGS_CURRENT: 	Amount drawing during the month of the previous credit<br>		
# AMT_DRAWINGS_OTHER_CURRENT: 	Amount of other drawings during the month of the previous credit<br>		
# AMT_DRAWINGS_POS_CURRENT: 	Amount drawing or buying goods during the month of the previous credit<br>		
# AMT_INST_MIN_REGULARITY: 	Minimal installment for this month of the previous credit<br>		
# AMT_PAYMENT_CURRENT: 	How much did the client pay during the month on the previous credit<br>		
# AMT_PAYMENT_TOTAL_CURRENT: 	How much did the client pay during the month in total on the previous credit<br>		
# AMT_RECEIVABLE_PRINCIPAL: 	Amount receivable for principal on the previous credit<br>		
# AMT_RECIVABLE: 	Amount receivable on the previous credit<br>		
# AMT_TOTAL_RECEIVABLE: 	Total amount receivable on the previous credit<br>		
# CNT_DRAWINGS_ATM_CURRENT: 	Number of drawings at ATM during this month on the previous credit<br>		
# CNT_DRAWINGS_CURRENT: 	Number of drawings during this month on the previous credit<br>		
# CNT_DRAWINGS_OTHER_CURRENT: 	Number of other drawings during this month on the previous credit<br>		
# CNT_DRAWINGS_POS_CURRENT: 	Number of drawings for goods during this month on the previous credit<br>		
# CNT_INSTALMENT_MATURE_CUM: 	Number of paid installments on the previous credit<br>		
# NAME_CONTRACT_STATUS: 	Contract status (active signed,...) on the previous credit<br>		
# SK_DPD: 	DPD (Days past due) during the month on the previous credit<br>		
# SK_DPD_DEF: 	DPD (Days past due) during the month with tolerance (debts with low loan amounts are ignored) of the previous credit<br>		
# </p></blockquote>

# In[19]:


# Last amended: 23rd October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing credit_card_balance dataset
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[20]:


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


# In[21]:


# 1.3
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[22]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[23]:


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


# In[24]:


# 2.1
pathToFolder = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToFolder)


# In[25]:


# 2.2 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# ## About the Dataset
# credit_card_balance: It is monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.

# In[27]:


# 2.3 Read the data
cc = pd.read_csv(
                'credit_card_balance.csv.zip',
                 nrows = num_rows
                )


# In[28]:


# 2.4
cc.shape      # (rows = 38,40,312, columns = 23)
cc.head()


# ## Feature explanations:
# MONTHS_BALANCE: Month of balance relative to application date (-1 means the freshest balance date)<br>
# AMT_BALANCE: 	    Balance during the month of previous credit<br>
# AMT_CREDIT_LIMIT_ACTUAL: 	Credit card limit during the month of the previous credit<br>
# AMT_DRAWINGS_ATM_CURRENT: 	Amount drawing at ATM during the month of the previous credit<br>
# AMT_DRAWINGS_CURRENT: 	Amount drawing during the month of the previous credit<br>
# AMT_DRAWINGS_OTHER_CURRENT: 	Amount of other drawings during the month of the previous credit<br>
# AMT_DRAWINGS_POS_CURRENT: 	Amount drawing or buying goods during the month of the previous credit<br>
# AMT_INST_MIN_REGULARITY: 	Minimal installment for this month of the previous credit<br>
# AMT_PAYMENT_CURRENT: 	How much did the client pay during the month on the previous credit<br>
# AMT_PAYMENT_TOTAL_CURRENT: 	How much did the client pay during the month in total on the previous credit<br>
# AMT_RECEIVABLE_PRINCIPAL: 	Amount receivable for principal on the previous credit<br>
# AMT_RECIVABLE: 	Amount receivable on the previous credit<br>
# AMT_TOTAL_RECEIVABLE: 	Total amount receivable on the previous credit<br>
# CNT_DRAWINGS_ATM_CURRENT: 	Number of drawings at ATM during this month on the previous credit<br>
# CNT_DRAWINGS_CURRENT: 	Number of drawings during this month on the previous credit<br>
# CNT_DRAWINGS_OTHER_CURRENT: 	Number of other drawings during this month on the previous credit<br>
# CNT_DRAWINGS_POS_CURRENT: 	Number of drawings for goods during this month on the previous credit<br>
# CNT_INSTALMENT_MATURE_CUM: 	Number of paid installments on the previous credit<br>
# NAME_CONTRACT_STATUS: 	Contract status (active signed,...) on the previous credit<br>
# SK_DPD: 	DPD (Days past due) during the month on the previous credit<br>
# SK_DPD_DEF: 	DPD (Days past due) during the month with tolerance (debts with low loan amounts are ignored) of the previous credit<br>
# 

# In[29]:


# 2.5 There is one 'object' feature
cc.dtypes.value_counts()


# In[30]:


# 2.6 Transform the 'object' feature to OHE
cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)


# In[31]:


# 2.7
cc.shape   # (3840312, 30)
cat_cols   # Even NaN is a feature


# In[32]:


# 2.8 Drop this unique ID. We do not need it
cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)


# In[33]:


# 3.0 Aggregate all features over SK_ID_CURR.
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])


# In[34]:


# 3.1
cc_agg.shape     # (103558, 140)
cc_agg.head()    # It has multi-index feature


# In[35]:


# 3.2 Change column names
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])


# In[36]:


# 3.3 Create another feature
#     For each client, how many observations
#     exist in this dataset

cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()


# In[37]:


# 3.3.1
cc_agg['CC_COUNT'].head()


# In[38]:


# 4.0 Save the results for subsequent use:
cc_agg.to_csv("processed_creditCard_agg.csv.zip", compression = "zip")


# In[ ]:


################

