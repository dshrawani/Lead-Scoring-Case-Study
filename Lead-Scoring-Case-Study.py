#!/usr/bin/env python
# coding: utf-8

# ## Lead-Scoring Case Study

# ### Problem Statement
# #### An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses.
# 
# #### The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%.
# 
# #### There are a lot of leads generated in the initial stage, but only a few of them come out as paying customers. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion.
# 
# #### X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# ## Goals of the Case Study
# #### Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# In[467]:


#importing necessary libraries

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[333]:


#loading and reading data
leads_data = pd.read_csv('Leads.csv')
leads_data.head()


# In[334]:


#checking data


# In[335]:


leads_data.shape


# In[336]:


leads_data.info()


# In[337]:


leads_data.describe()


# In[338]:


#checking for duplicates
leads_data.duplicated(subset = ['Prospect ID'], keep = False).sum()


# In[339]:


leads_data.duplicated(subset = ['Lead Number'], keep = False). sum()


# ### No duplicate values found in Prospect ID and Lead Number
# 
# ### Clearly Prospect ID & Lead Number are two variables that are just indicative of the ID number of the Contacted People & can be dropped.
# 

# ## Exploratory Data Analysis:

# ### Data Cleaning and Treatment:

# In[340]:


#dropping Lead Number and Prospect ID since they have all unique values

leads_data.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[341]:


#Converting 'Select' values to NaN.

leads_data = leads_data.replace('Select', np.nan)


# In[342]:


leads_data.nunique()


# In[343]:


# Dropping unique valued columns

leads_data= leads_data.drop(['Magazine','Receive More Updates About Our Courses','I agree to pay the amount through cheque','Get updates on DM Content','Update me on Supply Chain Content'],axis=1)


# In[344]:


#checking for null values in each rows

leads_data.isnull().sum()


# In[345]:


# checking for % of null value in each row
round(100*(leads_data.isnull().sum())/len(leads_data.index),2)


# In[346]:


##dropping columns with more than 45% missing values

leads_data = leads_data.drop(['Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Activity Index','Lead Profile','Lead Quality','How did you hear about X Education',],axis =1)


# In[347]:


leads_data.shape


# In[348]:


#checking null values percentage after dropping 
round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)


# In[349]:


###There is a huge value of null variables in some columns as seen above. But removing the rows with the null value will cost us a lot of data and they are important columns. So, instead we are going to replace the NaN values with 'not provided'. This way we have all the data and almost no null values. In case these come up in the model, it will be of no use and we can drop it off then.


# In[350]:


leads_data['Specialization'] = leads_data['Specialization'].fillna('not provided')
leads_data['City'] = leads_data['City'].fillna('not provided')
leads_data['Tags'] = leads_data['Tags'].fillna('not provided')
leads_data['What matters most to you in choosing a course'] = leads_data['What matters most to you in choosing a course'].fillna('not provided')
leads_data['What is your current occupation'] = leads_data['What is your current occupation'].fillna('not provided')
leads_data['Country'] = leads_data['Country'].fillna('not provided')
leads_data.info()


# In[351]:


#checking null values percentage

round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)


# In[352]:


leads_data.shape


# ## Categorical Veriables Analysis:
# 

# In[353]:


leads_data['Country'].value_counts()


# In[354]:


#creating a function for further analysis


# In[355]:


def slots(x):
    category = ""
    if x == "India":
        category = "India"
    elif x == "not provided":
        category = "not provided"
    else:
        category = "outside india"
    return category

leads_data['Country'] = leads_data.apply(lambda x:slots(x['Country']), axis = 1)
leads_data['Country'].value_counts()


# In[356]:


# Since India is the most common occurence among the non-missing values we can impute all not provided values with India

leads_data['Country'] = leads_data['Country'].replace('not provided','India')
leads_data['Country'].value_counts()


# In[357]:


# Checking the percent of lose if the null values are removed
round(100*(sum(leads_data.isnull().sum(axis=1) > 1)/leads_data.shape[0]),2)


# In[358]:


leads_data = leads_data[leads_data.isnull().sum(axis=1) <1]


# In[359]:


# Rechecking the percentage of missing values
round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)


# In[360]:


leads_data.shape


# In[361]:


#plotting spread of Country columnn after replacing NaN values

plt.figure(figsize=(15,5))
s1=sns.countplot(leads_data.Country, hue=leads_data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# ### As we can see the Number of Values for India are quite high (nearly 97% of the Data), this column can be dropped

# In[362]:


#creating a list of columns to be droppped

cols_to_drop=['Country']


# In[363]:


#checking value counts of "City" column

leads_data['City'].value_counts(dropna=False)


# In[364]:


#plotting spread of City columnn

plt.figure(figsize=(10,5))
s1=sns.countplot(leads_data.City, hue=leads_data.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[365]:


plt.figure(figsize = (20,40))

plt.subplot(6,2,1)
sns.countplot(leads_data['Lead Origin'])
plt.title('Lead Origin')

plt.subplot(6,2,2)
sns.countplot(leads_data['Do Not Email'])
plt.title('Do Not Email')

plt.subplot(6,2,3)
sns.countplot(leads_data['Do Not Call'])
plt.title('Do Not Call')

plt.subplot(6,2,4)
sns.countplot(leads_data['Country'])
plt.title('Country')

plt.subplot(6,2,5)
sns.countplot(leads_data['Search'])
plt.title('Search')
plt.subplot(6,2,6)
sns.countplot(leads_data['Newspaper Article'])
plt.title('Newspaper Article')

plt.subplot(6,2,7)
sns.countplot(leads_data['X Education Forums'])
plt.title('X Education Forums')

plt.subplot(6,2,8)
sns.countplot(leads_data['Newspaper'])
plt.title('Newspaper')

plt.subplot(6,2,9)
sns.countplot(leads_data['Digital Advertisement'])
plt.title('Digital Advertisement')

plt.subplot(6,2,10)
sns.countplot(leads_data['Through Recommendations'])
plt.title('Through Recommendations')

plt.subplot(6,2,11)
sns.countplot(leads_data['A free copy of Mastering The Interview'])
plt.title('A free copy of Mastering The Interview')
plt.subplot(6,2,12)
sns.countplot(leads_data['Last Notable Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')


plt.show()


# In[366]:


sns.countplot(leads_data['Lead Source']).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[367]:


plt.figure(figsize = (20,30))
plt.subplot(2,2,1)
sns.countplot(leads_data['Specialization']).tick_params(axis='x', rotation = 90)
plt.title('Specialization')
plt.subplot(2,2,2)
sns.countplot(leads_data['What is your current occupation']).tick_params(axis='x', rotation = 90)
plt.title('Current Occupation')
plt.subplot(2,2,3)
sns.countplot(leads_data['What matters most to you in choosing a course']).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')
plt.subplot(2,2,4)
sns.countplot(leads_data['Last Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')
plt.show()


# In[368]:


sns.countplot(leads_data['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# ## Numerical Variables Analysis:

# In[369]:


plt.figure(figsize = (10,10))
plt.subplot(221)
plt.hist(leads_data['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(222)
plt.hist(leads_data['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(223)
plt.hist(leads_data['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show( )         


# ## Relating all the categorical variables to Converted

# In[370]:


plt.figure(figsize = (10,10))

plt.subplot(2,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(2,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[371]:


plt.figure(figsize=(10 ,5))
plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Do Not Call')
plt.show()


# In[372]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')

plt.subplot(1,2,2)
sns.countplot(x='Country', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Country')
plt.show()


# In[373]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In[374]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')

plt.subplot(1,2,2)
sns.countplot(x='Search', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Search')
plt.show()


# In[375]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper Article', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Newspaper Article')

plt.subplot(1,2,2)
sns.countplot(x='X Education Forums', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('X Education Forums')
plt.show()


# In[376]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Through Recommendations', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Through Recommendations')

plt.subplot(1,2,2)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('A free copy of Mastering The Interview')
plt.show()


# In[377]:


sns.countplot(x='Last Notable Activity', hue='Converted', data= leads_data).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')
plt.show()


# In[378]:


#To check the correlation among varibles
plt.figure(figsize=(10,5))
sns.heatmap(leads_data.corr())
plt.show()


# ### It is understandable from the above EDA that there are many elements that have very little data and so will be of less relevance to our analysis.

# ## Outlier

# In[379]:


numeric = leads_data[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# In[380]:


plt.figure(figsize = (5,5))
sns.boxplot(y=leads_data['TotalVisits'])
plt.show()


# In[381]:


sns.boxplot(y=leads_data['Total Time Spent on Website'])
plt.show()


# In[382]:


sns.boxplot(y=leads_data['Page Views Per Visit'])
plt.show()


# In[383]:


##We can see presence of outliers in TotalVisits


# In[384]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = leads_data.TotalVisits.quantile(0.99)
leads_data = leads_data[(leads_data.TotalVisits <= Q3)]
Q1 = leads_data.TotalVisits.quantile(0.01)
leads_data = leads_data[(leads_data.TotalVisits >= Q1)]
sns.boxplot(y=leads_data['TotalVisits'])
plt.show()


# ## Creating Dummy Variables

# In[385]:


#We can drop "Tags" ,As tags variable is generated by the sales sales team after the disscussion with student otherwise it will increase the model accuracy.


# In[386]:


#list of columns to be dropped
cols_to_drop=['Country','Tags']


# In[387]:


#dropping columns
leads_data = leads_data.drop(cols_to_drop,1)
leads_data.info()


# In[388]:


#getting a list of categorical columns

cat_cols= leads_data.select_dtypes(include=['object']).columns
cat_cols


# In[389]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(leads_data[['Lead Origin','Specialization' ,'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
# Add the results to the master dataframe
Lead_data_dum = pd.concat([leads_data, dummy], axis=1)
Lead_data_dum


# In[390]:


Lead_data_dum = Lead_data_dum.drop(['City','What is your current occupation_not provided','Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 'Specialization', 'Specialization_not provided','What is your current occupation','What matters most to you in choosing a course', 'Search','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview', 'Last Notable Activity'], 1)
Lead_data_dum


# ## Test-Train Split
# 

# In[391]:


#Import the required library
from sklearn.model_selection import train_test_split


# In[392]:


X = Lead_data_dum.drop(['Converted'], 1)
X.head()


# In[393]:


# Putting the target variable in y
y = Lead_data_dum['Converted']
y.head()


# In[394]:


# Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)


# In[395]:


# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler
# Scale the three numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# ## Model Building

# In[396]:


# Import 'LogisticRegression'
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

logreg = LogisticRegression()
rfe = RFE(estimator=logreg, n_features_to_select=20)
rfe.fit(X_train, y_train)


# In[397]:


# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[398]:


# Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]


# ### All the variables selected by RFE, next statistics part (p-values and the VIFs)
# 
# 

# In[399]:


# Selecting columns selected by RFE
X_train = X_train[col]


# In[400]:


# Importing statsmodels
import statsmodels.api as sm


# In[401]:


train_sm = sm.add_constant(X_train)
X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[402]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[403]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### The VIF values seem fine but some p-values are 99%. So removing ' What is your current occupation_Housewife','Last Notable Activity_Had a Phone Conversation'.

# In[404]:


X_train.drop(['What is your current occupation_Housewife','Last Notable Activity_Had a Phone Conversation'], axis = 1, inplace = True)


# In[405]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[406]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[407]:


X_train.drop('Page Views Per Visit', axis = 1, inplace = True)


# In[408]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[409]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### All the VIF values are good and all the p-values are below 0.05. So we can fix model.

# ## Creating Prediction

# In[410]:


# Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[411]:


# Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[412]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[413]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# ## Model Evaluation

# In[414]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[415]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[416]:


# Predicted        No         Yes
# Actual
# No              3498      417
# Yes             837      1541


# In[417]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[418]:


#That's around 82% accuracy with is a very good value


# In[419]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[420]:


# Calculating the sensitivity
TP/(TP+FN)


# In[421]:


# Calculating the specificity
TN/(TN+FP)


# #### With the current cut off as 0.5 we have around 82% accuracy, sensitivity of around 70% and specificity of around 88%.

# ## Optimise Cut off (ROC Curve)

# #### The previous cut off was randomely selected. Now we have to find the optimum one.

# In[422]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[423]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[424]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[425]:


#The area under ROC curve is 0.88 which is a very good value


# In[426]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[427]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[428]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[429]:


#From the graph it is visible that the optimal cut off is at 0.35.


# In[430]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()


# In[431]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[432]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[433]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[434]:


# Calculating the sensitivity
TP/(TP+FN)


# In[435]:


# Calculating the specificity
TN/(TN+FP)


# #### With the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%
# 
# 

# ## Prediction on Test set

# In[436]:


#Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[437]:


col = X_train.columns


# In[438]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[439]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[440]:


# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final


# In[441]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[442]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[443]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[444]:


# Calculating the sensitivity
TP/(TP+FN)
0.7923604309500489


# In[445]:


# Calculating the specificity
TN/(TN+FP)


# #### With the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%
# 
# 

# ## Precision-Recall

# In[446]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[447]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[448]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# #### With the current cut off as 0.35 we have Precision around 79% and Recall around 70%
# 
# 

# ## Precision and recall tradeoff

# In[449]:


from sklearn.metrics import precision_recall_curve


# In[450]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[451]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[452]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[453]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()


# In[454]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[455]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[456]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[457]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[458]:


#Recall = TP / TP + FN
TP / (TP + FN)


# #### With the current cut off as 0.44 we have Precision around 76% and Recall around 76.3% and accuracy 82 %.

# ## Prediction on Test set

# In[459]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[460]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)
y_pred_final


# ## Check the overall accuracy
# 
# 

# In[461]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[462]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[463]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[464]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[465]:


#Recall = TP / TP + FN
TP / (TP + FN)


# #### With the current cut off as 0.41 we have Precision around 75% , Recall around 73% and accuracy 80.5%.

# #### The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model.

# ## Conclusion

# ### It was found that the variables that mattered the most in the potential buyers are (In descending order) :

# ## - TotalVisits
# ## - The total time spend on the Website.
# ## - Lead Origin_Lead Add Form
# ## - Lead Source_Direct Traffic
# ## - Lead Source_Google
# ## - Lead Source_Welingak Website
# ## - Lead Source_Organic Search
# ## - Lead Source_Referral Sites
# ## - Lead Source_Welingak Website
# ## - Do Not Email_Yes
# ## - Last Activity_Email Bounced
# ## - Last Activity_Olark Chat Conversation

# ### Keeping these in mind the X Education can flourish as they have a very high chance to get almost all the potential buyers to change their mind and buy their courses.

# In[ ]:




