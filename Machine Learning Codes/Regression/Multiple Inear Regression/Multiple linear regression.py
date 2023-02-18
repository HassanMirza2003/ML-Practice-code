import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.2 , random_state=0)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(xtrain , ytrain)

y_pred = regresor.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))

#evaluating Algo performance using rscore
from sklearn.metrics import r2_score
r =r2_score(ytest , y_pred)
print(r)

