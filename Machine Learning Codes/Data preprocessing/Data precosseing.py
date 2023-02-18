#importing the Libraires 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 

#importing data sets 
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values
#print(x)
#print(y)

#Taking care of missing data 
from sklearn.impute import SimpleImputer as si 
imputer = si(missing_values=np.nan , strategy='mean')
imputer.fit(x[: , 1:3] )  #excluding the String row only selecting the data with numercial value
x[: , 1:3] = imputer.transform(x[: , 1:3]) #Replacing the empty values with the new values 
print(x)

#encoding the categorial data.

 #for independent Variable 
from sklearn.compose import ColumnTransformer as ctf 
from sklearn.preprocessing import OneHotEncoder as oht
ct = ctf(transformers= [('encoders' , oht() , [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#print(x)

#for Dependent Variable 
from sklearn.preprocessing import LabelEncoder as lb
le = lb()
y = le.fit_transform(y)
#print(y)
#print(dataset)
#splitting into Train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y, test_size = 0.2 , random_state= 1)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train[:, 3:] = ss.fit_transform[x_train[: , 3:]]
x_test[:, 3:] = ss.transform[x_test[: , 3:]]





