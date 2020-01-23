
# coding: utf-8

# In[7]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')


# ## Loading Data

# In[8]:

data = pd.read_csv('data/cs-training.csv', index_col=0)
data.head()


# In[9]:

target_col = 'SeriousDlqin2yrs'


# ## Splitting into Training and Dev Sets

# In[10]:

train = data.sample(n=120000, random_state=3)
dev = data[~data.index.isin(train.index)]
train[target_col].value_counts(normalize=True), dev[target_col].value_counts(normalize=True)


# ## Cleaning Data

# In[11]:

def clean_data(df, income_mv, dependent_mv):
    """
    Imputes missing values
    Floors & ceils the outliers
    Combines smaller categories into a single category
    """
    
    df.columns = [col.replace('-', '_') for col in df.columns]
    
    # RevolvingUtilizationOfUnsecuredLines
    df.loc[df['RevolvingUtilizationOfUnsecuredLines'] >1, 'RevolvingUtilizationOfUnsecuredLines'] = 1
    
    # age
    df.loc[df['age']< 21, 'age'] = 21
    
    # NumberOfTime30-59DaysPastDueNotWorse
    df.loc[df['NumberOfTime30_59DaysPastDueNotWorse']>=2, 'NumberOfTime30_59DaysPastDueNotWorse'] = 2
    
    # NumberOfTime60-89DaysPastDueNotWorse
    df.loc[df['NumberOfTime60_89DaysPastDueNotWorse']>=1, 'NumberOfTime60_89DaysPastDueNotWorse'] = 1
    
    # NumberOfTimes90DaysLate
    df.loc[df['NumberOfTimes90DaysLate']>=1, 'NumberOfTimes90DaysLate'] = 1
    
    # MonthlyIncome
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_mv) #replace with median
    
    # NumberOfOpenCreditLinesAndLoans

    # NumberRealEstateLoansOrLines
    df.loc[df['NumberRealEstateLoansOrLines']>=3, 'NumberRealEstateLoansOrLines'] = 3
    
    # NumberOfDependents
    df.loc[df['NumberOfDependents']>=3, 'NumberOfDependents'] = 3
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(dependent_mv) #replace with median
    
    return df



# In[12]:

train = clean_data(train, 5400, 0)
dev = clean_data(dev, 5400, 0)


# In[13]:

def get_bins_from_training_data(df):
    '''bin variables based on the eda done in the other notebook'''
    binsdf = {}
    _, binsdf['RevolvingUtilizationOfUnsecuredLines'] = pd.qcut(df['RevolvingUtilizationOfUnsecuredLines'], 5, retbins=True, labels=None)
    _, binsdf['DebtRatio'] = pd.qcut(df['DebtRatio'], 5, retbins=True, labels=None)
    _, binsdf['age'] = pd.qcut(df['age'], 5, retbins=True, labels=None)
    _, binsdf['MonthlyIncome'] = pd.qcut(df['MonthlyIncome'], 4, retbins=True, labels=None)
    _, binsdf['NumberOfOpenCreditLinesAndLoans'] = pd.qcut(df['NumberOfOpenCreditLinesAndLoans'], 5, retbins=True, labels=None)
    return binsdf

binsdf = get_bins_from_training_data(train)


# In[21]:

def create_bins(df, binsdf):
    '''create bins on the unseen data set'''
    retdf = pd.DataFrame()
    for column in df.columns:
        if column in binsdf.keys():
            retdf[column] = pd.cut(df[column], bins=binsdf[column], labels=False, include_lowest=True)
        else:
            retdf[column] = df[column].copy(deep=True)
    return retdf

def create_bins_2(df, binsdf):
    '''create bins on the unseen data set'''
    retdf = pd.DataFrame()
    for column in df.columns:
        if column in binsdf.keys():
            retdf[column] = pd.cut(df[column], bins=binsdf[column], include_lowest=True)
        else:
            retdf[column] = df[column].copy(deep=True)
    return retdf

train_binned_df = create_bins(train, binsdf)
dev_binned_df = create_bins(dev, binsdf)

train_binned_df_2 = create_bins_2(train, binsdf)
dev_binned_df_2 = create_bins_2(dev, binsdf)


# In[22]:

def weight_of_evidence(df, labelcol, col, categorical_col=None):
    '''calculates weight of evidence values for each category in a given variable (col)'''
    if categorical_col is None:
        categorical_col = col
    
    tempdf = df.groupby(by=categorical_col).agg({labelcol:{
                                                        '_counts': 'size',
                                                        '_bads': lambda x: len(x[x==1]),
                                                        '_goods': lambda x: len(x[x==0])
                                                    }})
    tempdf.columns  = [col+ column for column in tempdf.columns.droplevel(0)]
    tempdf[col+'_distri_tot'] = tempdf[col+'_counts']/(tempdf[col+'_counts'].sum())
    tempdf[col+'_distri_bads'] = tempdf[col+'_bads']/(tempdf[col+'_bads'].sum())
    tempdf[col+'_distri_goods'] = tempdf[col+'_goods']/(tempdf[col+'_goods'].sum())
    tempdf[col+'_bad_rate'] = tempdf[col+'_bads']/tempdf[col+'_counts']
    tempdf[col+'_woe'] = np.log(tempdf[col+'_distri_goods']) - np.log(tempdf[col+'_distri_bads'])
    return tempdf


def information_value(df, col):
    '''calculates Information Value using the Weight of evidence scores'''
    df['diff_col'] = df[col+'_distri_goods'] - df[col+'_distri_bads']
    return np.sum(df['diff_col']*df[col+'_woe'])


# In[23]:

def get_woe_from_training_data(binned_df):
    '''calculates the wWeight of Evidence and Information Values based on the training data set for all the features'''
    woe_dict = {}
    iv_dict = {}
    for column in binned_df.columns:
        if column == target_col:
            pass
        else:
            woe_df = weight_of_evidence(binned_df, target_col, column)
            woe_column = [col for col in woe_df.columns if 'woe' in col][0]
            woe_dict[column] = dict(woe_df[woe_column])
            iv_dict[column] = information_value(woe_df, column)
    return woe_dict, iv_dict
 

woe, iv = get_woe_from_training_data(create_bins(train, binsdf))


# In[24]:

iv = pd.DataFrame.from_dict(iv, orient='index')
iv.columns = ['information_value']
iv.sort_values(by='information_value', ascending=False)


# In[25]:

def create_woe(binned_df, woe):
    '''Applies the calculated WoE values on the unseen data set'''
    for column in woe.keys():
        binned_df[column] = binned_df[column].map(woe[column])
    return binned_df


train_woe = create_woe(train_binned_df, woe)
dev_woe =  create_woe(dev_binned_df, woe)


# In[26]:

train_woe.columns = [col.replace('-', '_') for col in train_woe.columns]
dev_woe.columns = [col.replace('-', '_') for col in dev_woe.columns]
train_woe.columns


# ## Model

# In[27]:

from sklearn.linear_model import LogisticRegression


# In[111]:

var_cols = [
     'RevolvingUtilizationOfUnsecuredLines',
            'age',
       'NumberOfTime30_59DaysPastDueNotWorse',
            'DebtRatio', 
            'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines',
            'NumberOfTime60_89DaysPastDueNotWorse',
       'NumberOfDependents'
           ]
target_col = 'SeriousDlqin2yrs'


# ### Fitting the model

# In[112]:

sk_model = LogisticRegression()
sk_model.fit(train_woe[var_cols], train_woe[target_col])


# In[113]:

sk_model.coef_


# In[114]:

#coefficients

coef = pd.DataFrame()
coef['feature'] = var_cols
coef['coef'] = sk_model.coef_[0]
coef


# We remove the NumberOfOpenCreditLinesAndLoans as it is not significant by p-value (0.9) and make the model again

# In[115]:

# saving model
# sk_model.save('model_sk_10_variables_1.pkl')


# ### Checking performance

# In[116]:

import sklearn.metrics as setrics
plt.style.use('dark_background')

def get_metrics(df, model, var_cols, label_col):
    '''plots roc-auc and calcualtes somers-d score'''
    roc_auc = setrics.roc_auc_score(df[label_col],  model.predict_proba(df[var_cols])[:,1])
    print('ROC_AUC: ', roc_auc)
    
    somersd = 2*roc_auc -1
    print('Somersd: ', somersd)
    
    fpr, tpr, _ = setrics.roc_curve(df[label_col],  model.predict_proba(df[var_cols])[:,1])
    plt.plot(fpr,tpr,label="data 1, auc="+str(roc_auc))
    plt.legend(loc=4)
    plt.show()

print('-----   Training Set Performance   --------\n')    
get_metrics(train_woe, sk_model, var_cols, label_col='SeriousDlqin2yrs')


# In[117]:

print('-----   Dev Set Performance   --------\n')    

get_metrics(dev_woe.dropna(), sk_model, var_cols, label_col='SeriousDlqin2yrs')


# ## Scorecard

# ### Calculating Offset
# 
# Let's set the odds of 50:1 at a score of 600

# In[47]:

score= 600
odds = 50
pdo = 20
factor = pdo/np.log(2)

offset = score - np.log(odds)*factor
offset


# In[52]:

def make_scorecard(woe_df, model, features, pdo=20, offset_score=600):
    '''calculates credit score'''
    factor = pdo/np.log(2)
    n = len(features)
    offset = offset_score - np.log(50)*factor
    
    scorecard = -(((woe_df[features]*(model.coef_)).sum(axis=1)) + model.intercept_)*factor +  offset
    return scorecard


# In[53]:

train_scorecard = make_scorecard(train_woe, sk_model, var_cols, offset_score=600)
dev_scorecard = make_scorecard(dev_woe, sk_model, var_cols, offset_score=600)


# In[54]:

train['credit_score'] = train_scorecard
train.to_csv('output/train_credit_scores.csv')

dev['credit_score'] = dev_scorecard
dev.to_csv('output/dev_credit_scores.csv')


# ## Predicting on Test Data

# In[118]:

test = pd.read_csv('data/cs-test.csv', index_col=0)
test.head()


# In[119]:

test = clean_data(test, 5400, 3)
test_binned_df = create_bins(test, binsdf)
test_woe = create_woe(test_binned_df, woe)
test_scorecard = make_scorecard(test_woe, sk_model, var_cols)
test_scorecard.describe()


# In[120]:

test['credit_score'] = test_scorecard
test.to_csv('output/test_credit_scores.csv')


# ### Score distribution

# In[121]:

dev['credit_score'] = dev_scorecard


# In[122]:

plt.style.use('dark_background')
dev['credit_bin'] = pd.qcut(dev['credit_score'], 10, labels=False)
dev['credit_bin_range'] = pd.qcut(dev['credit_score'], 10)
dev.groupby(by='credit_bin').agg('mean')[target_col].plot()


# In[123]:

score_bins = dev.groupby(by='credit_bin_range').agg('mean')[target_col].reset_index()
score_bins.columns =['credit_bin', 'default_rate']
score_bins['default_rate'] = score_bins['default_rate'].round(3).mul(100)
score_bins


# In[124]:

score_bins['min_score'] = score_bins['credit_bin'].apply(lambda x: int(x.left))
score_bins['max_score'] = score_bins['credit_bin'].apply(lambda x: int(x.right))
score_bins = score_bins.reset_index()
score_bins.columns = ['decile', 'credit_bin', 'default_rate', 'min_score', 'max_score']
score_bins = score_bins[['decile', 'credit_bin', 'min_score', 'max_score', 'default_rate']]
score_bins['decile'] = score_bins['decile'] + 1
score_bins


# In[125]:

dev_binned_df_3 = dev_binned_df_2.join(dev_woe, rsuffix='_woe')
dev_binned_df_3.head()


# In[127]:

score_list = []
dev_binned_df_3 = dev_binned_df_3.dropna()
for i, variable in enumerate(var_cols):
    for val in dev_binned_df_3[variable+'_woe'].unique():
#         try:
        temp_list = []
        temp_list.append(variable)
        _bin = dev_binned_df_3[dev_binned_df_3[variable+'_woe']==val][variable].mode().values[0]
#             if isinstance(_bin, Interval):
#                 _bin = (_bin.left, _bin.right)
        temp_list.append(_bin)
        temp_list.append((-val*sk_model.coef_[0][i] +                                     sk_model.intercept_/len(var_cols))*20/np.log(2) +                                     offset/len(var_cols))
        score_list.append(pd.Series(temp_list))
#         except:
#             pass
      


# In[128]:

score_df = pd.concat(score_list, axis=1).T
score_df.columns = ['variable', 'bin', 'score']
score_df['left'] = score_df['bin'].apply(lambda x:x if (isinstance(x, np.int64) or isinstance(x, np.float64)) else x.left)
some_df = score_df.sort_values(by=['variable', 'left'], ascending=True).drop(columns=['left'])
# some_df.to_csv('tt.csv')
some_df['score'] = some_df['score'].apply(lambda x: x[0].round(1))
some_df


# In[ ]:



