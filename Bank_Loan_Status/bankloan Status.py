# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:44:42 2020

@author: sumanth
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pickle

df=pd.read_csv('bankloan.csv')
df=df.dropna()
df.isnull().sum()

df=df.drop('Loan_ID',axis=1)
df['LoanAmount']=(df['LoanAmount']*1000).astype(int)
Counter(df['Loan_Status'])

pre_Y=df['Loan_Status']
pre_X=df.drop('Loan_Status',axis=1)
dn_X=pd.get_dummies(pre_X)
dn_X.columns
dn_Y=pre_Y.map(dict(Y=1,N=0))

smote=SMOTE(sampling_strategy='minority')
X1,y=smote.fit_sample(dn_X,dn_Y)
sc=MinMaxScaler()
X=sc.fit_transform(X1)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=45,shuffle=True)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

#X=pd.DataFrame(X_test)
#X.to_csv('H:\\data science\\DataSets\\X_test.csv')

classifier=Sequential()
classifier.add(Dense(400,activation='relu',kernel_initializer='random_normal',input_dim=X_test.shape[1]))
classifier.add(Dense(800,activation='relu',kernel_initializer='random_normal'))
classifier.add(Dense(10,activation='relu',kernel_initializer='random_normal'))
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=25,epochs=100,verbose=0)
eval_mode=classifier.evaluate(X_train,Y_train)
eval_mode


y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.52)

cm=confusion_matrix(y_pred,Y_test)
ax=plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

#labels,titles
ax.set_xlabel('predicted');ax.set_ylabel('Actual');
ax.set_title('confusion_matrix');
ax.xaxis.set_ticklabels(['NO','YES']);ax.yaxis.set_ticklabels(['NO','YES'])


from sklearn.model_selection import StratifiedKFold

xfold=StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
cvscores=[]
for train,test in xfold.split(X,y):
    model=Sequential()
    model.add(Dense(200,activation='relu',input_dim=17))
    model.add(Dense(400,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    # comppile
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X[train],y[train],epochs=100,verbose=0)
    
    scores=model.evaluate(X[test],y[test],verbose=0)
    print(model.metrics_names[1],scores[1]*100)
    cvscores.append(scores[1]*100)
print(np.mean(cvscores),np.std(cvscores))


