import numpy as np
import scipy
from sklearn import linear_model
import pandas as pd
from FUNCS import *



url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
path="FuelConsumption.csv"
try:
    file = open(path)
    file.close()
except FileNotFoundError:
    download(url, path)


df = pd.read_csv(path)

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

regr2 = linear_model.LinearRegression()
regr2.fit(x_train, y_train)
y_hat = regr2.predict(x_test)
print("COEF: ", regr2.coef_)
print("RSS: ", np.mean((y_hat - y_test) **2))
print("R SCORE: ", regr2.score(x_test, y_test))