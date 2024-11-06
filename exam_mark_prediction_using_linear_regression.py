##importing libraries
import pandas as pd
import numpy as np

##load the dataset
dataset=pd.read_csv("data.csv")

##dataset summary
# print(dataset.describe())
# print(dataset.head())

##Check null values
# print(dataset.isna().any())
# #outout -   
# hours        True
# age         False
# internet    False
# marks       False
# dtype: bool

##sloving null values
# dataset=dataset.dropna()
dataset.hours=dataset.hours.fillna(dataset.hours.mean().__format__('.2f'))
# print(dataset.head())

##segregating the into x and y
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

##Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

##Prediction
y_pred=regressor.predict(X_test)
print(np.column_stack((X_test,y_test,y_pred)))
##Accuracy
# from sklearn.metrics import accuracy_score
# accuracy=accuracy_score(y_test,y_pred)
# print("Accuracy is",accuracy)

##RMSE
from sklearn.metrics import mean_squared_error
rmse=mean_squared_error(y_test,y_pred)
print("RMSE is",rmse)

##Plotting the graph
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color='red') 
plt.show()