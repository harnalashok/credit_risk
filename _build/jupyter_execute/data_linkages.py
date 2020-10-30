#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk Competition
# 
# Consider this collection of notebooks as a case-study intended for those who are beginners in Machine Learning. We have tried to expand upon the code with our comments available in some of the notebooks on Kaggle. 
# 
# 
# # Data
# 
# The data as provided by [Home Credit](http://www.homecredit.net/about-us.aspx), is divided in seven interlinked tables. The data description is as follows. The interlinkages between data sources is depiced in the following diagram.
# 
# * application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature `SK_ID_CURR`. The training application data comes with the `TARGET` indicating 0: the loan was repaid or 1: the loan was not repaid. 
# * bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
# * bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length. 
# * previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature `SK_ID_PREV`. 
# * POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
# * credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
# * installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment. 
# 
# This diagram shows how all of the data is related:
# 
# ![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
# 
# The definitions of all the columns is provided in `HomeCredit_columns_description.csv`. 
# 
# 
# __Some references__
# 
# * [Credit Education](https://myscore.cibil.com/CreditView/creditEducation.page?enterprise=CIBIL&_ga=2.245893574.372615569.1603669858-164953316.1602941832&_gac=1.254345978.1602941832.CjwKCAjwrKr8BRB_EiwA7eFaplQtBsmINtLxLHOCalWYdx-uO20kyaj0AvRVD8WKNO4cj5mP7MoBTRoC6TEQAvD_BwE)
# 
# * [Credit Appraisal Methodology and Statndards](https://www.paisadukan.com/credit-assessment-methodology)
# 

# In[ ]:


#############

