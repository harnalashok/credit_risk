#!/usr/bin/env python
# coding: utf-8

# # Bureau and Bureau Balance data
# 
# <blockquote>*bureau.csv* data concerns client's earlier credits from other financial institutions. Some of the credits may be active and some are closed. Each previous (or ongoing) credit has its own row (only <u>one</u> row per credit) in *bureau* dataset. As a single client might have taken other loans from other financial institutions, for each row in the *application_train* data (ie *application_train.csv*) we can have multiple rows in this table. Feature explanations for this dataset are as below.</blockquote>

# ## Feature explanations
# ### Bureau table
# <blockquote><p style="font-size:13px">
# SK_ID_CURR: 	ID of loan in our sample - one loan in our sample can have 0,1,2 or more related previous credits in credit bureau <br>
# SK_BUREAU_ID: 	Recoded ID of previous Credit Bureau credit related to our loan (unique coding for each loan application)<br>
# CREDIT_ACTIVE: 	Status of the Credit Bureau (CB) reported credits<br>
# CREDIT_CURRENCY: 	Recoded currency of the Credit Bureau credit<br>
# DAYS_CREDIT: 	How many days before current application did client apply for Credit Bureau credit<br>
# CREDIT_DAY_OVERDUE: 	Number of days past due on CB credit at the time of application for related loan in our sample<br>
# DAYS_CREDIT_ENDDATE: 	Remaining duration of CB credit (in days) at the time of application in Home Credit<br>
# DAYS_ENDDATE_FACT: 	Days since CB credit ended at the time of application in Home Credit (only for closed credit)<br>
# AMT_CREDIT_MAX_OVERDUE: 	Maximal amount overdue on the Credit Bureau credit so far (at application date of loan in our sample)<br>
# CNT_CREDIT_PROLONG: 	How many times was the Credit Bureau credit prolonged<br>
# AMT_CREDIT_SUM: 	Current credit amount for the Credit Bureau credit<br>
# AMT_CREDIT_SUM_DEBT: 	Current debt on Credit Bureau credit<br>
# AMT_CREDIT_SUM_LIMIT: 	Current credit limit of credit card reported in Credit Bureau<br>
# AMT_CREDIT_SUM_OVERDUE: 	Current amount overdue on Credit Bureau credit<br>
# CREDIT_TYPE: 	Type of Credit Bureau credit (Car, cash,...)<br>
# DAYS_CREDIT_UPDATE: 	How many days before loan application did last information about the Credit Bureau credit come<br>
# AMT_ANNUITY: 	Annuity of the Credit Bureau credit<br>
#     </p></blockquote>
#     
# ### Bureau Balance table
# <blockquote>SK_BUREAU_ID:	Recoded ID of Credit Bureau credit (unique coding for each application) - use this to join to CREDIT_BUREAU table<br>
# MONTHS_BALANCE:	Month of balance relative to application date (-1 means the freshest balance date)	time only relative to the application<br>
# STATUS:	Status of Credit Bureau loan during the month<br> 	
# </blockquote>

# In[1]:


# Last amended: 21st October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing bureau and bureau_balance datasets.
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[60]:


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
#     (Code modified to
#      exclude 'category' dtype)
import reducing

# 1.2 Misc
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[61]:


# 1.3
pd.set_option('display.max_colwidth', -1)


# In[62]:


# 1.4 Display multiple commands outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[63]:


# 2.0 One-hot encoding function. Uses pd.get_dummies()
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


# In[64]:


# 3.0 Prepare to read data
pathToData = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToData)


# In[65]:


# 3.1 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# In[66]:


# 3.2 Read bureau data first
bureau = pd.read_csv(
                     'bureau.csv.zip',
                     nrows = None    # Read all rows
                    )

# 3.2.1 Reduce memory usage by appropriately
#       changing data-types per feature:

bureau = reducing.Reducer().reduce(bureau)


# In[67]:


# 3.2.2 Explore data now
bureau.head(5)
bureau.shape   # (rows:17,16,428, cols: 17)
bureau.dtypes


# In[68]:


# 3.2.3 In all, how many are categoricals?
bureau.dtypes.value_counts()


# In[69]:


# 3.3
bureau.shape                       # (1716428, 17)

# 3.3.1
# What is the actual number of persons
#  who might have taken multiple loans?

bureau['SK_ID_CURR'].nunique()     # 305811  -- Many duplicate values exist
                                   #            Consider SK_ID_CURR as Foreign Key
                                   #            Primary key exists in application_train data
                                   # Primary key: SK_ID_BUREAU
            
# 3.3.2
# As expected, there are no duplicate values here
bureau['SK_ID_BUREAU'].nunique()   # 1716428 -- Unique id for each row 


# In[70]:


# 3.4 Summary of active/closed cases from bureau
# We aggregate on these also
bureau['CREDIT_ACTIVE'].value_counts()


# ## Aggregation
# <blockquote><i>bureau_balance</i> will be aggregated and merged with <i>bureau</i>. <i>bureau</i> will then be aggregated and merged with <i>'application_train'</i> data. <i>bureau</i> will be aggregated in three different ways. This aggregation will be by <i>SK_ID_CURR</i>. Finally, aggregated <i>bureau</i>, called <i>bureau_agg</i>, will be merged with  <i>'application_train'</i> over (<i>SK_ID_CURR</i>).<br>
# Aggregation over time is one way to extract behaviour of client. All categorical data is first OneHotEncoded (OHE). What is unique about this OHE is that NaN values are treated as categories. 
# 

# In[71]:


# 4.0 OneHotEncode 'object' types in bureau
bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)


# In[72]:


# 4.1
bureau.head()
bureau.shape          # (1716428, 40); 17-->40
print(bureau_cat)     # List of added columns


# ## bureau_balance
# <blockquote>It is monthly data about the remaining balance of each one of the previous credits of clients that exist in dataset <i>bureau</i>. Each previous credit is identified by a unique ID, <i>SK_ID_BUREAU</i>, in dataset <i>bureau</i>. Each row in <i>bureau_balance</i> is one month of credit-due (from previous credit), and a single previous credit can have multiple rows, one for each month of the credit length.<br> In my personal view, it should be in decreasing order. That is, for every person identified by <i>SK_ID_BUREAU</i>, credits should be decreasing each passing month.</blockquote>

# In[74]:


# 5.0 Read over bureau_balance data
#     and reduce memory usage through
#     conversion of data-types:

bb = pd.read_csv('bureau_balance.csv.zip', nrows = None)
bb = reducing.Reducer().reduce(bb)


# In[75]:


# 5.0.1 Display few rows 
bb.head(10)

# 5.0.2 & Compare
bb.shape      # (27299925, 3) 
bureau.shape  # (1716428, 17)


# In[76]:


# 5.1 There is just one 'object' column
bb.dtypes.value_counts()


# In[77]:


# 5.2 Is the data about all bureau cases?
#      No, it appears it is not for all cases

bb['SK_ID_BUREAU'].nunique()    # 817395 << 1716428


# In[78]:


# 5.3 Just which cases are present in 'bureau' but absent
#     in 'bb'
bb_id_set = set(bb['SK_ID_BUREAU'])             # Set of IDs in bb
bureau_id_set = set(bureau['SK_ID_BUREAU'])     # Set of IDs in bureau


# In[79]:


# 5.4 And here is the difference list.
#      How many of them? 
list(bureau_id_set - bb_id_set)[:5]      # sample [6292791,6292792,6292793,6292795,6292796,6292797,6292798,6292799]
len(bureau_id_set - bb_id_set) # 942074


# In[80]:


# 5.5 OK. So let us OneHotEncode bb
bb, bb_cat = one_hot_encoder(bb, nan_as_category)


# In[81]:


# 5.6 Examine the results
bb.head()
bb.shape   # (27299925, 11) ; 3-->11
           # 1 (ID) + 1 (numeric) + 9 (dummy)
bb_cat     # New columns added


# ## Performing aggregations in bb
# <blockquote>There is one numeric feature: <i>'MONTHS_BALANCE'</i>. On this feature we will perform ['min', 'max', 'size']. And on the rest of the features,dummy features, we will perform [mean]. Aggregation is by unique bureau ID, <i>SK_ID_BUREAU</i>. Resulting dataset is called <i>bureau_agg</i>.</blockquote>  
# 

# In[82]:


# 6.0 Bureau balance: Perform aggregations and merge with bureau.csv
#     First prepare a dictionary listing operations to be performed
#     on various features:

bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
for col in bb_cat:
    bb_aggregations[col] = ['mean']

# 6.0.1    
len(bb_aggregations)     # 10  


# In[83]:


# 6.1 So what all aggregations to perform column-wise

bb_aggregations


# In[84]:


# 6.2 Perform aggregations now in bb:

grouped =  bb.groupby('SK_ID_BUREAU')
bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)


# In[85]:


# 6.3
bb_agg.shape      # (817395, 12)
bb_agg.columns

# 6.3.1 Note that 'SK_ID_BUREAU'
#       the grouping column is
#       now table-index

bb_agg.head()


# In[86]:


# 6.4 Rename bb_agg columns
bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])


# In[87]:


# 6.4.1
bb_agg.columns.tolist()
bb_agg.head()


# In[88]:


# 6.5 Merge aggregated bb with bureau

bureau = bureau.join(
                     bb_agg,
                     how='left',
                     on='SK_ID_BUREAU'
                    )


# In[89]:


# 6.5.1

bureau.head()
bureau.shape   # (1716428, 52)
bureau.dtypes  


# In[90]:


# 6.5.2 Just for curiosity, what happened
#       to those rows in 'bureau' where there
#       was no matching record in bb_agg. The list
#       of such IDs is:
#       [6292791,6292792,6292793,6292795,6292796,6292797,6292798,6292799]

bureau[bureau['SK_ID_BUREAU'] ==6292791]


# In[91]:


# 6.6 Drop SK_ID_BUREAU as bb has finally merged.

bureau.drop(['SK_ID_BUREAU'],
            axis=1,
            inplace= True
           )


# In[33]:


# We have three types of columns
# Categorical columns generated from bureau
# Categorical columns generated from bb
# Numerical columns


# ## Performing aggregations in bureau
# <blockquote>Aggregate 14 original numeric columns, as: ['min', 'max', 'mean', 'var']<br>
# Aggregate rest of the columns that is dummy columns as: [mean]. <br>
# This constitutes one of the three aggretaions. Aggregation is by <i>SK_ID_CURR</i>. Resulting dataset is called <i>bureau_agg</i></blockquote>

# In[92]:


# 7.0 Have a look at bureau again.
#      SK_ID_CURR repeats for many cases.
#         So, there is a case for aggregation

bureau.shape     # (1716428, 51)
bureau.head()


# In[93]:


## Aggregation strategy
# 7.1 Numeric features
#     Columns: Bureau + bureau_balance numeric features
#              Last three columns are from bureau_balance
#              Total: 11 + 3 = 14

num_aggregations = {
                     'DAYS_CREDIT':             ['min', 'max', 'mean', 'var'],
                     'DAYS_CREDIT_ENDDATE':     ['min', 'max', 'mean'],
                     'DAYS_CREDIT_UPDATE':      ['mean'],
                     'CREDIT_DAY_OVERDUE':      ['max', 'mean'],
                     'AMT_CREDIT_MAX_OVERDUE':  ['mean'],
                     'AMT_CREDIT_SUM':          ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_DEBT':     ['max', 'mean', 'sum'],
                     'AMT_CREDIT_SUM_OVERDUE':  ['mean'],
                     'AMT_CREDIT_SUM_LIMIT':    ['mean', 'sum'],
                     'AMT_ANNUITY':             ['max', 'mean'],
                     'CNT_CREDIT_PROLONG':      ['sum'],
                     'MONTHS_BALANCE_MIN':      ['min'],
                     'MONTHS_BALANCE_MAX':      ['max'],
                     'MONTHS_BALANCE_SIZE':     ['mean', 'sum']
                   }

len(num_aggregations)   # 14


# In[94]:


# 7.2 Bureau categorical features. Derived from:
#       'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 
#        Total: 

cat_aggregations = {}
bureau_cat      # bureau_cat are newly created dummy columns
                #  but all are numerical columns

# 7.2.1    
len(bureau_cat) # 26    


# In[95]:


# 7.2.2 For all these new dummy columns in bureau, we will
#       take mean
for cat in bureau_cat: cat_aggregations[cat] = ['mean']
cat_aggregations    

len(cat_aggregations)   # 26


# In[96]:


# 7.3.1 In addition, we have in bureau. columns that merged
#        from 'bb' ie bb_cat
#         So here is our full list
bb_cat
len(bb_cat)             # 9

# 7.3.2
for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
cat_aggregations 

len(cat_aggregations)   # 26 + 9 = 35


# In[97]:


# 7.4 Have a look at bureau columns again
#      Just to compare above results with what
#       already exists

bureau.columns        # 51
len(bureau.columns)   # 35 (dummy) + 14 (num) + 1 (SK_ID_CURR) + 1 (DAYS_ENDDATE_FACT) = 51


# In[98]:


# 7.5 Now that we have decided 
#     our aggregation strategy for each column
#      (except 2), let us now aggregate:
#         Note that SK_ID_CURR now becomes an index to data

grouped = bureau.groupby('SK_ID_CURR')
bureau_agg = grouped.agg({**num_aggregations, **cat_aggregations})


# In[99]:


# 7.6
bureau_agg.head()
bureau_agg.shape  # (305811, 62) (including newly created min, max etc columns)


# In[100]:


# 7.7 Remove hierarchical index from bureau_agg
bureau_agg.columns       # 62 
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])


# In[101]:


# 7.8
bureau_agg.head()
# 7.8.1 Note that SK_ID_CURR is now an index to table
bureau_agg.columns   # 62: Due to creation of min, max, var etc columns


# In[102]:


# 7.9 No duplicate index
bureau_agg.index.nunique()   # 305811
len(set(bureau_agg.index))   # 305811


# ## More Aggregation and merger
# <blockquote>We now filter <i>bureau</i> on <i>CREDIT_ACTIVE_Active</i> feature. This will create two subsets of data. This feature has values of 1 and 0.<br> Filter data where <i>CREDIT_ACTIVE_Active</i> value is 1. Then aggregate(only) numeric features of this filtered data-subset by grouping on <i>SK_ID_CURR</i>. Next, filter, bureau, on <i>CREDIT_ACTIVE_Closed = 1 </i>. And again aggregate the subset on numeric features. Merge all these with <i>bureau_agg</i> (NOT <i>bureau.</i>)<br><br> It is as if we are trying to extract the behaviour of those whose credits are active and those whose credits are closed.</blockquote>

# In[103]:


# 8.0 In which cases credit is active? Filter data
active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
active.head()
active.shape   # (630607, 51)


# In[104]:


# 8.1 Aggregate numercial features of the filtered subset over SK_ID_CURR
active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)


# In[105]:


# 8.1.1
active_agg.head()
active_agg.shape   # (251815, 27)


# In[106]:


# 8.1.2 Rename multi-indexed columns
active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
active_agg.columns


# In[107]:


# 9.0 Difference between length of two datasets
active_agg_set = set(active_agg.index)
bureau_agg_set = set(bureau_agg.index)
len(bureau_agg_set)     # 305811
len(active_agg_set)     # 251815
list(bureau_agg_set - active_agg_set)[:4]   # Few examples: {131074, 393220, 262149, 262153]


# In[108]:


# 9.1 Merge bureau_agg with active_agg over 'SK_ID_CURR'
bureau_agg = bureau_agg.join(
                             active_agg,
                             how='left',
                             on='SK_ID_CURR'
                             )


# In[109]:


# 9.2 
bureau_agg.shape    # (305811, 89)


# In[110]:


# 9.3 Obviouly some rows will hold NaN values for merged columns
bureau_agg.loc[[131074,393220,262149, 262153]]


# In[111]:


# 9.4 Release memory
del active, active_agg
gc.collect()


# In[112]:


# 10.0 Same steps for the  CREDIT_ACTIVE_Closed =1 cases
#     Bureau: Closed credits - using only numerical aggregations
closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')


# In[113]:


# 10.1
bureau_agg.shape   # (305811, 116)


# In[114]:


# 10.2
del closed, closed_agg, bureau
gc.collect()


# In[115]:


# 10.3 SK_ID_CURR is index. Index is also saved by-default.
bureau_agg.to_csv("processed_bureau_agg.csv.zip", compression = "zip")


# In[ ]:


##################

