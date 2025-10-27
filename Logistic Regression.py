#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-27T13:51:15.596Z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Titanic_train.csv')

df.describe()

df.info()

df.head()

df['Age'].fillna(df['Age'].median(),inplace=True)

df.drop('Cabin',axis=1,inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

df.isnull().sum()

df.head()

df.duplicated().sum()

le=LabelEncoder()

df['Sex']=le.fit_transform(df['Sex'])

df=pd.get_dummies(df,columns=['Embarked'],drop_first=True,dtype=int)

df.drop('Name',axis=1,inplace=True)

df['Tickets']=df['Ticket'].apply(lambda x: str(x).split()[-1])
df['Tickets']=pd.to_numeric(df['Tickets'],errors='coerce')

df.drop('Ticket',axis=1,inplace=True)

df

df.columns

corr=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,cmap='coolwarm',annot=True)

from scipy.stats.mstats import winsorize

from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.5)
outlier = iso.fit_predict(df.select_dtypes(include='number'))
df = df[outlier == 1]

for col in df.select_dtypes(include='number').columns:
    df[col]=winsorize(df[col],limits=[0.05,0.05])

df.boxplot()
plt.show()

num_col=df.select_dtypes(include='number').columns
df[num_col].hist(figsize=(8,6),bins=30,color='teal')
plt.tight_layout()
plt.show()

std=StandardScaler()

df[['Age','Fare','Tickets']]=std.fit_transform(df[['Age','Fare','Tickets']])

df.drop('PassengerId',axis=1,inplace=True)

df['SibSp']=df['SibSp'].astype(int)

df.drop(['Parch','Embarked_Q'],axis=1,inplace=True)

df.head()

df['Tickets'].fillna(df['Tickets'].mean(),inplace=True)

target = df['Survived']
feature = df.drop(['Survived'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(feature,target,train_size=0.75,random_state=150)
print(x_train.shape)
print(y_train.shape)
print(x_train.shape)
print(y_test.shape)

model =LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)

y_proc = model.predict_proba(x_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test,y_proc)

roc_auc= auc(fpr,tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color='blue',label='ROC curve(AUC=%0.2f)'%roc_auc)
plt.grid()
plt.plot([0,1],[0,1],color='gray')
plt.title('ROC Curve Logistic Regression Model')
plt.xlabel('ROC Curve of fpr')
plt.ylabel('ROC Curve of tpr')

 import pickle

file='my_model.pkl'

pickle.dump(model,open(file,'wb'))

file

# * Precision and recall
# * well both are the metrix and the precision the prpoportion of correctly predicted positive cases out of all postive predictions made by the model
# * Recall on the other hand measure the proportion of the actual positive cases that were correctly identified by the model
# * And is metrix used for binary classification and its especialy used for imbalanced dataset


# * cross_validation
# * cross_validation is the model on multiple different parts of the dataset
# * k-flod split data into k part,rotate test/training
# * And is used for avoid overfiting test general performance
# * And is best for small datasets,binary classification,imbalanced data