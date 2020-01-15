import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm, model_selection
from sklearn.model_selection import cross_validate

print("ok")

df = quandl.get("WIKI/AMZN")
print(df.tail())

df = df[['Adj. Close']]
forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# Training
# clf = LinearRegression()
# clf.fit(X_train,y_train)
# # Testing
# confidence = clf.score(X_test, y_test)
# print("confidence: ", confidence)

