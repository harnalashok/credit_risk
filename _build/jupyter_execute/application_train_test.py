#!/usr/bin/env python
# coding: utf-8

# # Process train and test files

# ## About data
# <blockquote><i>application_train/application_test</i>: This is main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature <i>SK_ID_CURR</i>. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. Feature descriptions are as below. These have been taken from file <i>'HomeCredit_columns_description.csv'>/i>. For a more accurate description, please refer to the file.</blockquote> 

# ## Feature descriptions
# <blockquote><p style="font-size:13px">
# SK_ID_CURR: 	ID of loan in our sample<br>
# TARGET: 	Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)<br>
# NAME_CONTRACT_TYPE: 	Identification if loan is cash or revolving<br>
# CODE_GENDER: 	Gender of the client<br>
# FLAG_OWN_CAR: 	Flag if the client owns a car<br>
# FLAG_OWN_REALTY: 	Flag if client owns a house or flat<br>
# CNT_CHILDREN: 	Number of children the client has<br>
# AMT_INCOME_TOTAL: 	Income of the client<br>
# AMT_CREDIT: 	Credit amount of the loan<br>
# AMT_ANNUITY: 	Loan annuity<br>
# AMT_GOODS_PRICE: 	For consumer loans it is the price of the goods for which the loan is given<br>
# NAME_TYPE_SUITE: 	Who was accompanying client when he was applying for the loan<br>
# NAME_INCOME_TYPE: 	Clients income type (businessman, working, maternity leave,…)<br>
# NAME_EDUCATION_TYPE: 	Level of highest education the client achieved<br>
# NAME_FAMILY_STATUS: 	Family status of the client<br>
# NAME_HOUSING_TYPE: 	What is the housing situation of the client (renting, living with parents, ...)<br>
# REGION_POPULATION_RELATIVE: 	Normalized population of region where client lives (higher number means the client lives in more populated region)<br>
# DAYS_BIRTH: 	Client's age in days at the time of application<br>
# DAYS_EMPLOYED: 	How many days before the application the person started current employment<br>
# DAYS_REGISTRATION: 	How many days before the application did client change his registration<br>
# DAYS_ID_PUBLISH: 	How many days before the application did client change the identity document with which he applied for the loan<br>
# OWN_CAR_AGE: 	Age of client's car<br>
# FLAG_MOBIL: 	Did client provide mobile phone (1=YES, 0=NO)<br>
# FLAG_EMP_PHONE: 	Did client provide work phone (1=YES, 0=NO)<br>
# FLAG_WORK_PHONE: 	Did client provide home phone (1=YES, 0=NO)<br>
# FLAG_CONT_MOBILE: 	Was mobile phone reachable (1=YES, 0=NO)<br>
# FLAG_PHONE: 	Did client provide home phone (1=YES, 0=NO)<br>
# FLAG_EMAIL: 	Did client provide email (1=YES, 0=NO)<br>
# OCCUPATION_TYPE: 	What kind of occupation does the client have<br>
# CNT_FAM_MEMBERS: 	How many family members does client have<br>
# REGION_RATING_CLIENT: 	Our rating of the region where client lives (1,2,3)<br>
# REGION_RATING_CLIENT_W_CITY: 	Our rating of the region where client lives with taking city into account (1,2,3)<br>
# WEEKDAY_APPR_PROCESS_START: 	On which day of the week did the client apply for the loan<br>
# HOUR_APPR_PROCESS_START: 	Approximately at what hour did the client apply for the loan<br>
# REG_REGION_NOT_LIVE_REGION: 	Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)<br>
# REG_REGION_NOT_WORK_REGION: 	Flag if client's permanent address does not match work address (1=different, 0=same, at region level)<br>
# LIVE_REGION_NOT_WORK_REGION: 	Flag if client's contact address does not match work address (1=different, 0=same, at region level)<br>
# REG_CITY_NOT_LIVE_CITY: 	Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)<br>
# REG_CITY_NOT_WORK_CITY: 	Flag if client's permanent address does not match work address (1=different, 0=same, at city level)<br>
# LIVE_CITY_NOT_WORK_CITY: 	Flag if client's contact address does not match work address (1=different, 0=same, at city level)<br>
# ORGANIZATION_TYPE: 	Type of organization where client works<br>
# EXT_SOURCE_1: 	Normalized score from external data source<br>
# EXT_SOURCE_2: 	Normalized score from external data source<br>
# EXT_SOURCE_3: 	Normalized score from external data source<br>
# APARTMENTS_AVG: 	Normalized information about building where the client lives.<br>
# BASEMENTAREA_AVG: 	Normalized information about building where the client lives.<br>
# YEARS_BEGINEXPLUATATION_AVG: 	Normalized information about building where the client lives.<br>
# YEARS_BUILD_AVG: 	Normalized information about building where the client lives.<br>
# COMMONAREA_AVG: 	Normalized information about building where the client lives.<br>
# ELEVATORS_AVG: 	Normalized information about building where the client lives.<br>
# ENTRANCES_AVG: 	Normalized information about building where the client lives.<br>
# FLOORSMAX_AVG: 	Normalized information about building where the client lives.<br>
# FLOORSMIN_AVG: 	Normalized information about building where the client lives.<br>
# LANDAREA_AVG: 	Normalized information about building where the client lives.<br>
# LIVINGAPARTMENTS_AVG: 	Normalized information about building where the client lives.<br>
# LIVINGAREA_AVG: 	Normalized information about building where the client lives.<br>
# NONLIVINGAPARTMENTS_AVG: 	Normalized information about building where the client lives.<br>
# NONLIVINGAREA_AVG: 	Normalized information about building where the client lives.<br>
# APARTMENTS_MODE: 	Normalized information about building where the client lives.<br>
# BASEMENTAREA_MODE: 	Normalized information about building where the client lives.<br>
# YEARS_BEGINEXPLUATATION_MODE: 	Normalized information about building where the client lives.<br>
# YEARS_BUILD_MODE: 	Normalized information about building where the client lives<br>
# COMMONAREA_MODE: 	Normalized information about building where the client lives<br>
# ELEVATORS_MODE: 	Normalized information about building where the client lives<br>
# ENTRANCES_MODE: 	Normalized information about building where the client lives<br>
# FLOORSMAX_MODE: 	Normalized information about building where the client lives<br>
# FLOORSMIN_MODE: 	Normalized information about building where the client lives<br>
# LANDAREA_MODE: 	Normalized information about building where the client lives<br>
# LIVINGAPARTMENTS_MODE: 	Normalized information about building where the client lives<br>
# LIVINGAREA_MODE: 	Normalized information about building where the client lives<br>
# NONLIVINGAPARTMENTS_MODE: 	Normalized information about building where the client lives<br>
# NONLIVINGAREA_MODE: 	Normalized information about building where the client lives<br>
# APARTMENTS_MEDI: 	Normalized information about building where the client lives<br>
# BASEMENTAREA_MEDI: 	Normalized information about building where the client lives<br>
# YEARS_BEGINEXPLUATATION_MEDI: 	Normalized information about building where the client lives<br>
# YEARS_BUILD_MEDI: 	Normalized information about building where the client lives<br>
# COMMONAREA_MEDI: 	Normalized information about building where the client lives<br>
# ELEVATORS_MEDI: 	Normalized information about building where the client lives<br>
# ENTRANCES_MEDI: 	Normalized information about building where the client lives<br>
# FLOORSMAX_MEDI: 	Normalized information about building where the client lives<br>
# FLOORSMIN_MEDI: 	Normalized information about building where the client lives<br>
# LANDAREA_MEDI: 	Normalized information about building where the client lives<br>
# LIVINGAPARTMENTS_MEDI: 	Normalized information about building where the client lives<br>
# LIVINGAREA_MEDI: 	Normalized information about building where the client lives<br>
# NONLIVINGAPARTMENTS_MEDI: 	Normalized information about building where the client lives<br>
# NONLIVINGAREA_MEDI: 	Normalized information about building where the client lives<br>
# FONDKAPREMONT_MODE: 	Normalized information about building where the client lives<br>
# HOUSETYPE_MODE: 	Normalized information about building where the client lives<br>
# TOTALAREA_MODE: 	Normalized information about building where the client lives<br>
# WALLSMATERIAL_MODE: 	Normalized information about building where the client lives<br>
# EMERGENCYSTATE_MODE: 	Normalized information about building where the client lives<br>
# OBS_30_CNT_SOCIAL_CIRCLE: 	How many observation of client's social surroundings with observable 30 DPD (days past due) default<br>
# DEF_30_CNT_SOCIAL_CIRCLE: 	How many observation of client's social surroundings defaulted on 30 DPD (days past due) <br>
# OBS_60_CNT_SOCIAL_CIRCLE: 	How many observation of client's social surroundings with observable 60 DPD (days past due) default<br>
# DEF_60_CNT_SOCIAL_CIRCLE: 	How many observation of client's social surroundings defaulted on 60 (days past due) DPD<br>
# DAYS_LAST_PHONE_CHANGE: 	How many days before application did client change phone<br>
# FLAG_DOCUMENT_2: 	Did client provide document 2<br>
# FLAG_DOCUMENT_3: 	Did client provide document 3<br>
# FLAG_DOCUMENT_4: 	Did client provide document 4<br>
# FLAG_DOCUMENT_5: 	Did client provide document 5<br>
# FLAG_DOCUMENT_6: 	Did client provide document 6<br>
# FLAG_DOCUMENT_7: 	Did client provide document 7<br>
# FLAG_DOCUMENT_8: 	Did client provide document 8<br>
# FLAG_DOCUMENT_9: 	Did client provide document 9<br>
# FLAG_DOCUMENT_10: 	Did client provide document 10<br>
# FLAG_DOCUMENT_11: 	Did client provide document 11<br>
# FLAG_DOCUMENT_12: 	Did client provide document 12<br>
# FLAG_DOCUMENT_13: 	Did client provide document 13<br>
# FLAG_DOCUMENT_14: 	Did client provide document 14<br>
# FLAG_DOCUMENT_15: 	Did client provide document 15<br>
# FLAG_DOCUMENT_16: 	Did client provide document 16<br>
# FLAG_DOCUMENT_17: 	Did client provide document 17<br>
# FLAG_DOCUMENT_18: 	Did client provide document 18<br>
# FLAG_DOCUMENT_19: 	Did client provide document 19<br>
# FLAG_DOCUMENT_20: 	Did client provide document 20<br>
# FLAG_DOCUMENT_21: 	Did client provide document 21<br>
# AMT_REQ_CREDIT_BUREAU_HOUR: 	Number of enquiries to Credit Bureau about the client one hour before application<br>
# AMT_REQ_CREDIT_BUREAU_DAY: 	Number of enquiries to Credit Bureau about the client one day before application<br>
# AMT_REQ_CREDIT_BUREAU_WEEK: 	Number of enquiries to Credit Bureau about the client one week before applicationion)<br>
# AMT_REQ_CREDIT_BUREAU_MON: 	Number of enquiries to Credit Bureau about the client one month before application <br>
# AMT_REQ_CREDIT_BUREAU_QRT: 	Number of enquiries to Credit Bureau about the client 3 month before application <br>
# AMT_REQ_CREDIT_BUREAU_YEAR: 	Number of enquiries to Credit Bureau about the client one day year<br></p></blockquote>
# 

# In[82]:


# Last amended: 24rd October, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\home_credit_default_risk
# Objective: 
#           Solving Kaggle problem: Home Credit Default Risk
#           Processing application train/test datasets
#
# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# Ref: https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features


# In[83]:


# 1.0 Libraries

import numpy as np
import pandas as pd

# 1.1 Reduce read data size
#     There is a file reducing.py
#      in this folder. A class
#       in it is used to reduce
#        dataframe size
#     (Code modified to
#      exclude 'category' dtype)
#     Refer: https://wkirgsn.github.io/2018/02/10/auto-downsizing-dtypes/
import reducing

# 1.2 Misc
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[84]:


# 1.3 In view of large dataset, some useful options
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[85]:


# 1.4 Display outputs from multiple commands from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[86]:


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


# In[87]:


# 2.1
pathToFolder = "C:\\Users\\Administrator\\OneDrive\\Documents\\home_credit_default_risk"
os.chdir(pathToFolder)


# In[88]:


# 2.2 Some constants
num_rows=None                # Implies read all rows
nan_as_category = True       # While transforming 
                             #   'object' columns to dummies


# In[89]:


# 3.0 Read previous application data first
df = pd.read_csv(
                   'application_train.csv.zip',
                   nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

df = reducing.Reducer().reduce(df)


# In[90]:


# 3.0 Read previous application data first
test_df = pd.read_csv(
                      'application_test.csv.zip',
                       nrows = num_rows
                   )

# 3.0.1 Reduce memory usage by appropriately
#       changing data-types per feature:

test_df = reducing.Reducer().reduce(test_df)


# In[91]:


# 3.1
df.shape   # (307511, 122)
df.head()


# In[66]:


# 3.1.1 There are 16 object types
df.dtypes.value_counts()


# In[67]:


# 3.2
test_df.shape   # (48744, 121)
test_df.head()


# In[68]:


# 3.3 There are 16 object types
test_df.dtypes.value_counts()


# In[69]:


# 3.4 Append test_df to train
df = df.append(test_df).reset_index()


# In[70]:


# 3.5 Examine merged data
df.shape     # (356255, 123)
df.head()


# In[71]:


# 3.6 This gender is rare. So such
#     rows can be dropped
df[df['CODE_GENDER'] == 'XNA']


# In[72]:


# 3.7 Optional: Remove 4 applications with XNA CODE_GENDER (train set)
df = df[df['CODE_GENDER'] != 'XNA']


# In[73]:


# 3.8 Categorical features with Binary encode (0 or 1; two categories)
for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    df[bin_feature], uniques = pd.factorize(df[bin_feature])


# In[74]:


# 3.8.1
df.head()
uniques


# In[75]:


# 3.8.2
df.dtypes
df.dtypes.value_counts()
df.dtypes['CODE_GENDER']
df.dtypes['FLAG_OWN_CAR']
df.dtypes['FLAG_OWN_REALTY']


# In[76]:


# 4.0 Categorical features with One-Hot encode
df, cat_cols = one_hot_encoder(df, nan_as_category)


# In[77]:


# 4.1
len(cat_cols)   # 146
cat_cols


# In[78]:


# 4.2 NaN values for DAYS_EMPLOYED: 365.243 -> nan
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)


# In[79]:


# 4.3 Some simple new features (percentages)
df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']


# In[81]:


# 5.0 Save the results for subsequent use:
df.to_csv("processed_df.csv.zip", compression = "zip")


# In[ ]:


#################

