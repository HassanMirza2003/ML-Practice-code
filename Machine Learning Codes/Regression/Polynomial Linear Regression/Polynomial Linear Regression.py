import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[:, -1].values

#training on linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#training on Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x) 
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#visualize in Linear regression
plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x) , color = 'blue' )
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualize in poynomial Regression
plt.scatter(x,y , color = 'red')
plt.plot(x, lin_reg2.predict(x_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))