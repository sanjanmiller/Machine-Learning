#Data Preprocessing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r'C:\Users\HP\Downloads\Salary_Data.csv')
x = dataset.iloc[:, :-1]  # Years of experience (Independent variable)
y = dataset.iloc[:, -1] # Salary (Dependent variable)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

#Model Building
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
plt.scatter(x_test, y_test, color = 'red')  # Real salary data (training)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
m_slope=regressor.coef_
print(m_slope)
c_inter=regressor.intercept_
print(c_inter)
y_15=m_slope*15+c_inter #predicting salary for 15 year experience(y=mx+c)
print(y_15)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

#Statistics for ML
dataset.mean()
dataset['Salary'].mean()
dataset.median()
dataset['Salary'].mode()
dataset.describe()
dataset.var()
dataset.std()
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])
dataset.skew()
dataset['Salary'].skew()
dataset.sem()

#z-score
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)
mean_total = np.mean(dataset.values)# here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataset.values-mean_total)**2)
print(SST)
r_square = 1-SSR/SST
print(r_square)
bias=regressor.score(x_train,y_train)
print(bias)
variance=regressor.score(x_test,y_test)
print(variance)



import pickle
filename = 'regressor.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as regressor.pkl")
import os
print(os.getcwd())


