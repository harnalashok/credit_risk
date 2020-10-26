#!/usr/bin/env python
# coding: utf-8

# # Previous Applications
# ## About the data
# <blockquote>previous_application: This dataset has details of previous applications made by clients to Home Credit. Only those clients find place here who also exist in <i>application</i> data. Each current loan in the <i>application</i> data (identified by <i>SK_ID_CURR</i>) can have multiple previous loan applications. Each previous application has one row and is identified by the feature <i>SK_ID_PREV</i>.</blockquote> 
# 

# ## Feature Explanations
# <blockquote><p style="font-size:13px"> 
# SK_ID_PREV : 	ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loan applications in Home Credit, previous application could, but not necessarily have to lead to credit) <br>						
# SK_ID_CURR: 	ID of loan in our sample<br>						
# NAME_CONTRACT_TYPE: 	Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application<br>						
# AMT_ANNUITY: 	Annuity of previous application<br>						
# AMT_APPLICATION: 	For how much credit did client ask on the previous application<br>						
# AMT_CREDIT: 	Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT<br>						
# AMT_DOWN_PAYMENT: 	Down payment on the previous application<br>						
# AMT_GOODS_PRICE: 	Goods price of good that client asked for (if applicable) on the previous application<br>						
# WEEKDAY_APPR_PROCESS_START: 	On which day of the week did the client apply for previous application<br>						
# HOUR_APPR_PROCESS_START: 	Approximately at what day hour did the client apply for the previous application<br>						
# FLAG_LAST_APPL_PER_CONTRACT: 	Flag if it was last application for the previous contract. Sometimes by mistake of client or our clerk there could be more applications for one single contract<br>						
# NFLAG_LAST_APPL_IN_DAY: 	Flag if the application was the last application per day of the client. Sometimes clients apply for more applications a day. Rarely it could also be error in our system that one application is in the database twice<br>						
# NFLAG_MICRO_CASH: 	Flag Micro finance loan<br>						
# RATE_DOWN_PAYMENT: 	Down payment rate normalized on previous credit<br>						
# RATE_INTEREST_PRIMARY: 	Interest rate normalized on previous credit<br>						
# RATE_INTEREST_PRIVILEGED: 	Interest rate normalized on previous credit<br>						
# NAME_CASH_LOAN_PURPOSE: 	Purpose of the cash loan<br>						
# NAME_CONTRACT_STATUS: 	Contract status (approved, cancelled, ...) of previous application<br>						
# DAYS_DECISION: 	Relative to current application when was the decision about previous application made<br>						
# NAME_PAYMENT_TYPE: 	Payment method that client chose to pay for the previous application<br>						
# CODE_REJECT_REASON: 	Why was the previous application rejected<br>						
# NAME_TYPE_SUITE: 	Who accompanied client when applying for the previous application<br>						
# NAME_CLIENT_TYPE: 	Was the client old or new client when applying for the previous application<br>						
# NAME_GOODS_CATEGORY: 	What kind of goods did the client apply for in the previous application<br>						
# NAME_PORTFOLIO: 	Was the previous application for CASH, POS, CAR, …<br>						
# NAME_PRODUCT_TYPE: 	Was the previous application x-sell o walk-in<br>						
# CHANNEL_TYPE: 	Through which channel we acquired the client on the previous application<br>						
# SELLERPLACE_AREA: 	Selling area of seller place of the previous application<br>						
# NAME_SELLER_INDUSTRY: 	The industry of the seller<br>						
# CNT_PAYMENT: 	Term of previous credit at application of the previous application<br>						
# NAME_YIELD_GROUP: 	Grouped interest rate into small medium and high of the previous application<br>						
# PRODUCT_COMBINATION: 	Detailed product combination of the previous application<br>						
# DAYS_FIRST_DRAWING: 	Relative to application date of current application when was the first disbursement of the previous application<br>						
# DAYS_FIRST_DUE: 	Relative to application date of current application when was the first due supposed to be of the previous application<br>						
# DAYS_LAST_DUE_1ST_VERSION: 	Relative to application date of current application when was the first due of the previous application<br>						
# DAYS_LAST_DUE: 	Relative to application date of current application when was the last due date of the previous application<br>						
# DAYS_TERMINATION: 	Relative to application date of current application when was the expected termination of the previous application<br>						
# NFLAG_INSURED_ON_APPROVAL: 	Did the client requested insurance during the previous application<br>	</p></blockquote>

# In[1]:


# Last amended: 24rd October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing previous_application dataset
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[39]:


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


# In[40]:


# 1.3
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)


# In[41]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[42]:


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


# In[43]:


# 2.1
pathToFolder = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToFolder)


# In[44]:


# 2.2 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# In[45]:


# 3.0 Read previous application data first
prev = pd.read_csv(
                   'previous_application.csv.zip',
                   nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

prev = reducing.Reducer().reduce(prev)


# In[46]:


# 3.0.2
prev.shape             # (rows=16,70,214, cols = 37)
prev.head(5)
prev.columns


# In[47]:


# 3.1 Let us examine how many unique IDs exist 

prev['SK_ID_PREV'].nunique()   # 1670214 Unique number
prev['SK_ID_CURR'].nunique()   # 338857  So a number of repeat exist
                               # We have to aggregate over it
                               #  to extract behaviour of clients


# In[48]:


# 3.2 Let us see distribution of dtypes
#     There are 16 'object' types here

prev.dtypes.value_counts()


# In[49]:


# 3.3
prev.shape                       # (1670214, 37)

# 3.3.1
# What is the actual number of persons
#  who might have taken multiple loans?

prev['SK_ID_CURR'].nunique()     # 338857  -- Many duplicate values exist
                                   #            Consider SK_ID_CURR as Foreign Key
                                   #            Primary key exists in application_train data
                                   # Primary key: SK_ID_BUREAU
            
# 3.3.2
# As expected, there are no duplicate values here

prev['SK_ID_PREV'].nunique()   # 1670214 -- Unique id for each row 


# In[50]:


# 4.0 OneHotEncode (OHE) 'object' types in bureau

prev, cat_cols = one_hot_encoder(
                                 prev,
                                 nan_as_category= True
                                 )


# In[52]:


# 4.1

len(cat_cols)      # 159
cat_cols


# In[53]:


# 4.2.1 Just examine NULLs in few features
prev['DAYS_FIRST_DRAWING'].isnull().sum()     # 673065
# 4.2.2 And also this special constant value: 365243
(prev['DAYS_FIRST_DRAWING'] == 365243).sum()  # 934444

prev['DAYS_FIRST_DUE'].isnull().sum()         # 673065
(prev['DAYS_FIRST_DUE'] == 365243).sum()      #  40645

prev['DAYS_LAST_DUE'].isnull().sum()          # 673065
(prev['DAYS_LAST_DUE'] == 365243).sum()       # 211221

prev['DAYS_TERMINATION'].isnull().sum()       # 673065
(prev['DAYS_TERMINATION']== 365243).sum()     # 225913


# In[54]:


# 4.3 Examine total number of unique values
#     in each one of the above four features

prev['DAYS_FIRST_DRAWING'].nunique()     # 2838
prev['DAYS_FIRST_DRAWING'].sort_values(ascending = False)[:5]
prev['DAYS_FIRST_DUE'].nunique()         # 2892
prev['DAYS_LAST_DUE'].nunique()          # 2873
prev['DAYS_TERMINATION'].nunique()       # 2830


# In[55]:


# 4.4 Convert Days 365243 values to nan

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


# In[56]:


# 4.5 So how many NULLS now exist in each one of
#     these four features:

prev['DAYS_FIRST_DRAWING'].isnull().sum()     # 1607509
prev['DAYS_FIRST_DUE'].isnull().sum()         #  713710
prev['DAYS_LAST_DUE'].isnull().sum()          #  884286
prev['DAYS_TERMINATION'].isnull().sum()       #  898978


# ## Perform aggregations
# <blockquote>On the whole of dataset, perform aggregations for numerical features and perform aggregations on just created OHE features. Numerical features are being aggregated as: <i>min, max, mean..</i> while OHE features aggregation is just <i>'mean'</i>.</blockquote>

# In[57]:


# 5.0 One special feature
#     Add feature: value ask / value received percentage

prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

# 5.1 Numeric features aggregations:
#     Dictionary of what all operations are to 
#     performed on numerical features:

num_aggregations = {
                     'AMT_ANNUITY':             ['min', 'max', 'mean'],
                     'AMT_APPLICATION':         ['min', 'max', 'mean'],
                     'AMT_CREDIT':              ['min', 'max', 'mean'],
                     'APP_CREDIT_PERC':         ['min', 'max', 'mean', 'var'],
                     'AMT_DOWN_PAYMENT':        ['min', 'max', 'mean'],
                     'AMT_GOODS_PRICE':         ['min', 'max', 'mean'],
                     'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                     'RATE_DOWN_PAYMENT':       ['min', 'max', 'mean'],
                     'DAYS_DECISION':           ['min', 'max', 'mean'],
                     'CNT_PAYMENT':             ['mean', 'sum'],
                    }


# In[58]:


# 5.2 Categorical features
#     Create a dictionary for aggregation operations:

cat_aggregations = {}
for cat in cat_cols:
    cat_aggregations[cat] = ['mean']

# 5.2.1    
cat_aggregations    


# In[59]:


# 5.3 Perform aggregation now on SK_ID_CURR:

grouped = prev.groupby('SK_ID_CURR')
prev_agg=grouped.agg({**num_aggregations, **cat_aggregations})


# In[60]:


# 5.3.1
prev_agg.shape    # (338857, 189)
prev_agg.columns
prev_agg.head()


# In[61]:


# 5.4 Rename multiindex columns:

prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])


# In[62]:


# 5.5
prev_agg.shape      # (338857, 189)
prev_agg.columns
prev_agg.head()


# ## More aggregations
# <blockquote>Table, <i>prev_agg</i>, from previous operations is our main table that we will carry to next exercise. To this aggregated table, we add more aggregations. <br>We will perform aggregations on two subsets of data. On both the subsets on numerical features only. One subset is extracted by setting <i>NAME_CONTRACT_STATUS_Approved == 1</i> and the other subset is extracted by setting <i>NAME_CONTRACT_STATUS_Refused == 1</i>.<br><br>It is as if we are trying to extract the behaviour of those whose previous applications have been approved and those whose previous applications have NOT been approved.</blockquote>
# 

# In[67]:


# 6.0 Previous Applications: Summarise numerical features from Approved Applications

approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)


# In[68]:


# 6.1 Look at the aggregated results:

approved_agg.columns
approved_agg.head()


# In[69]:


# 6.2 Rename multi-index column names:

approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])


# In[70]:


# 6.2.1 Look at it again:

approved_agg.shape     # (337698, 30)
approved_agg.head()


# In[71]:


# 6.3 Join 'approved_agg' with 'prev_agg'.

prev_agg = prev_agg.join(                      # prev_agg is on the left
                         approved_agg,         # table on the right
                         how='left',           # Join on left table. All its rows remain
                         on='SK_ID_CURR'       # Joining key. 
                        )


# In[72]:


# 6.3.1

prev_agg.shape     # (338857, 219)
prev_agg.head()


# In[73]:


# 6.4 Similarly for refused applications perform aggregations of numerical features:

refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)


# In[74]:


# 6.4.1

refused_agg.shape      # (118277, 30)
refused_agg.head()


# In[75]:


# 6.5

refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
refused_agg.head()
refused_agg.shape   # (118277, 30)


# In[76]:


# 7.0 Join refused_agg with prev_agg:

prev_agg = prev_agg.join(                     # prev_agg: left
                         refused_agg,         # table on the right
                         how='left',
                         on='SK_ID_CURR'
                        )


# In[77]:


# 7.1 Our final table:

prev_agg.shape     # 338857, 249)
prev_agg.head()


# In[78]:


# 8.0 Save the results for subsequent use:
prev_agg.to_csv("processed_prev_agg.csv.zip", compression = "zip")


# In[ ]:


####################

