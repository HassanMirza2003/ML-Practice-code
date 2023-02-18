import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Case  study : Predicting a Purchase using Age and Salaray of the person
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25 , random_state=0)

#fearure scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train , y_train)

print(classifier.predict(sc.transform([[30,87000]]))) #Where 30 is the Age and 87000 is the salary



