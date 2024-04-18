# loanModel.py file is to be used to create the loan decision XGBoost model 
# it will be called in the testMenu.py file to use the model to predict 
# someone's loan acceptance or rejection 
# code source: https://www.projectpro.io/article/loan-prediction-using-machine-learning-project-source-code/632#mcetoc_1gbmktaonv


import numpy as np
import pandas as pd 

import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier 

# feature explainers 
import lime 
from lime.lime_tabular import LimeTabularExplainer
# possible do import xai 
# https://contextual-ai.readthedocs.io/en/latest/tutorials/explainer/tutorial_lime_tabular_explainer.html

import joblib

loan_train = pd.read_csv('./datasets/train_csv.csv')
#dataset from kaggle https://www.kaggle.com/datasets/shaijudatascience/loan-prediction-practice-av-competition/ 
#print(loan_train.shape) #(614, 13)
#loan_train.head()

total_null = loan_train.isnull().sum().sort_values(ascending=False)
#total_null.head(10)


#replacing null values with the mode in loan_train becuase if we remove the null values 
#it will significantly decrease the size of the dataset so we use mode in this case as 
#most columns are binary 
loan_train['Gender'] = loan_train['Gender'].fillna( loan_train['Gender'].dropna().mode().values[0])
loan_train['Married'] = loan_train['Married'].fillna( loan_train['Married'].dropna().mode().values[0])
loan_train['Dependents'] = loan_train['Dependents'].fillna( loan_train['Dependents'].dropna().mode().values[0])
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna( loan_train['Self_Employed'].dropna().mode().values[0])
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna( loan_train['LoanAmount'].dropna().mean())
#use mean for LoanAmount instead of mean because it is not one of the categorical variable 
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna( loan_train['Loan_Amount_Term'].dropna().mode().values[0])
loan_train['Credit_History'] = loan_train['Credit_History'].fillna( loan_train['Credit_History'].dropna().mode().values[0])

#loan_train.info()
#fillna() method fills empty fields with whatever paramenter is given 
#dropna() will return the column values after removing the NULL values. Calculating the mean 
# or mode of this array of values and passing it to fillna() completes the process of filling nulls

'''print(set(loan_train['Gender'].values.tolist()))
print(set(loan_train['Dependents'].values.tolist()))
print(set(loan_train['Married'].values.tolist()))
print(set(loan_train['Education'].values.tolist()))
print(set(loan_train['Self_Employed'].values.tolist()))
print(set(loan_train['Loan_Status'].values.tolist()))
print(set(loan_train['Property_Area'].values.tolist()))'''
#categorical variables 
#looking at the unique values each of these non-numerical variables hold 

#map categorical variables to binary alternatives, for yes and no ( yes mapped to 1, no mapped to 0)
#for non-binary values like property area use get_dummies from pandas to automatically
#one-hot encode the variables, gender will be two dummy columns, dummy_male, dummy_female
loan_train['Loan_Status'] = loan_train['Loan_Status'].map({'N':0, 'Y':1}).astype(int)
loan_train = pd.get_dummies(loan_train, columns=['Gender', 'Dependents', 'Married', 
'Education', 'Self_Employed', 'Property_Area'])
#HERE WHEN REMOVING GET DUMMIES IT CHANGES THE OUTPUT OF THE LOAN DECISION, DECIDE IF WE SHOULD 
# (changes from credit-history being most impactful to self employed)
# MANUALLY MAP OUT THESE VALUES OR JUST HAVE THEM REMAIN LIKE THIS FOR MORE ACCURATE LIME EXPLANATIONS
# get_dummies uses one hot encoding to transform categorical into continuous (representing variables)
standardScaler = StandardScaler()
columnsToScale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
loan_train[columnsToScale] = standardScaler.fit_transform(loan_train[columnsToScale])

#loan_train.info()

# TRAIN TEST SPLIT OF DATA
#using train_test_split from sklearn and split ratio of 80:20 creat the train and test datasets
#POSSIBLY TRY REMOVING y_df and just having y_df as y and remove the y = y_df.values variable
y_df = loan_train['Loan_Status']
y = y_df.values
x_df = loan_train.drop(['Loan_Status', 'Loan_ID'], axis = 1)
x = x_df.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

type(x)

# MODEL DEFINITION, COMPILE, FIT
#XGBoost options for estimators 
gbm_param_grid = {
    'n_estimators': range(1,1000, 10),
    'max_depth': range(1,20),
    'learning_rate': [.1, .2, .3, .4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1],
}
# MODEL DEFINITION
xgb_classifier = XGBClassifier()
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=xgb_classifier, scoring= "accuracy", verbose = 0, n_iter= 100, cv =5)

#MODEL FIT
xgb_random.fit(x_train, y_train)
#print(f'Best parameters:  {xgb_random.best_params_}')

# MODEL PREDICTION
y_pred = xgb_random.predict(x_test)
print(f'Accuracy: {np.sum(y_pred==y_test)/len(y_test)}')
print(y_pred)

joblib.dump(xgb_random.best_estimator_, "loan_model.sav")

loan_approval_decision2 = joblib.load('loan_model.sav')

#importing the stored model from disk
#loan_approval_decision2 = joblib.load('loan_approval_decision.joblib')
#xgb_random.save_model('xgb_model.json')

#loaded_model = XGBClassifier()
#loaded_model.load_model('xgb_model.json')

#to understand feature importance 
def feature_imp(nump,model):
    feat = pd.DataFrame(columns=['feature', 'importance'])
    df = pd.DataFrame(nump, columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                                          'Credit_History', 'Gender_Female', 'Gender_Male', 'Dependents_0',
                                          'Dependents_1','Dependents_2','Dependents_3+',
                                          'Married_No', 'Married_Yes', 'Education_Graduate', 'Education_Not Graduate',
                                          'Self_Employed_No', 'Self_Employed_Yes', 'Property_Area_Rural',
                                          'Property_Area_Semiurban', 'Property_Area_Urban'])
    '''df = pd.DataFrame(nump, columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                                          'Credit_History', 'Gender', 'Dependents','Married', 'Education',
                                          'Self_Employed', 'Property_Area'])'''
    feat["feature"] = df.columns
    feat["importance"] = model.feature_importances_
    return feat.sort_values(by="importance", ascending=False)

imp_features = feature_imp(x_train, loan_approval_decision2)
#imp_features = feature_imp(x_train, loaded_model)
print(imp_features)

'''# feature_imp(x_train, xgb_random).plot('feature', 'importance', 'barh',
# figsize=(10,7), legend=False, ) --> FOR CHART WITH NO VALUES

df = feature_imp(x_train, loan_approval_decision2)

# Create the barh plot
ax = df.plot(kind='barh', x='feature', y='importance', figsize=(10, 7), legend=False)

# Add value labels to each bar
for bar in ax.patches:
    # The text to display (which is the y-value of each bar)
    text = f'{bar.get_width():.2f}'  # Format to 2 decimal places

    # x-position: bar.get_width()
    # y-position: bar.get_y() + bar.get_height() / 2
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2, text, 
            va='center', ha='right', fontsize=8)

# Show the plot
#plt.show()'''

# FUNCTION TO CONVERT THE TESTING DATASET INTO THE PROPER DATATYPES, REMOVE NULL VALUES, ETC 
def convert_data(loan_data):
    loan_data['Gender'] = loan_data['Gender'].fillna( loan_data['Gender'].dropna().mode().values[0])
    loan_data['Married'] = loan_data['Married'].fillna( loan_data['Married'].dropna().mode().values[0])
    loan_data['Dependents'] = loan_data['Dependents'].fillna( loan_data['Dependents'].dropna().mode().values[0])
    loan_data['Self_Employed'] = loan_data['Self_Employed'].fillna( loan_data['Self_Employed'].dropna().mode().values[0])
    loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna( loan_data['LoanAmount'].dropna().mean())
#use mean for LoanAmount instead of mean because it is not one of the categorical variable 
    loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna( loan_data['Loan_Amount_Term'].dropna().mode().values[0])
    loan_data['Credit_History'] = loan_data['Credit_History'].fillna( loan_data['Credit_History'].dropna().mode().values[0])
    
    #loan_data['Loan_Status'] = loan_data['Loan_Status'].map({'N':0, 'Y':1}).astype(int)
    loan_data = pd.get_dummies(loan_data, columns=['Gender', 'Dependents', 'Married', 
    'Education', 'Self_Employed', 'Property_Area'])

    #HERE WHEN REMOVING GET DUMMIES IT CHANGES THE OUTPUT OF THE LOAN DECISION, DECIDE IF WE SHOULD 
    # MANUALLY MAP OUT THESE VALUES OR JUST HAVE THEM REMAIN LIKE THIS FOR MORE ACCURATE LIME EXPLANATIONS
    '''loan_data['Gender'] = loan_data['Gender'].map({'Male': 0, 'Female':1}).astype(int)
    loan_data['Dependents'] = loan_data['Dependents'].map({'0':0, '1': 1, '2':2, '3+':3}).astype(int)
    loan_data['Married'] = loan_data['Married'].map({'No': 0, 'Yes':1}).astype(int)
    loan_data['Education'] = loan_data['Education'].map({'Graduate': 0, 'Not Graduate':1}).astype(int)
    loan_data['Self_Employed'] = loan_data['Self_Employed'].map({'No': 0, 'Yes':1}).astype(int)
    loan_data['Property_Area'] = loan_data['Property_Area'].map({'Urban': 2, 'Semiurban':1, 'Rural':0}).astype(int)'''
    
    standardScaler = StandardScaler()
    columnsToScale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    loan_data[columnsToScale] = standardScaler.fit_transform(loan_data[columnsToScale])
    #loan_data = loan_data.drop('Loan_ID', axis=1)
    return loan_data

def predict_loan_decision(dataset):
    #dataset_np= dataset.values 
    prediction_array = loan_approval_decision2.predict(dataset)
    #prediction_array = loaded_model.predict(dataset)
    return prediction_array[:]



