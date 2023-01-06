

# __________________ Importing Necessary Libararies __________________

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math


# Importing the dataset

data = pd.read_csv("Fish.csv")


# Checking the dataset if there are null values and printing the dataset columns in order to become 
# more familiar with the dataset 

print(data.isna().sum())
print(data.columns)


# Printing correlation among dataset columns in order to see if there is relation between any coulmn
corr = data[data.columns].corr() 
print(corr)


# Specifying and reshaping the features and target from dataset in order to be 
# ready for fitting our LinearRegression model
# We are assigning Width of the fish as feature for predicting the Weight of the fish
features = data['Width']
target = data['Weight']

x = np.array(features).reshape(-1, 1)
y = np.array(target)


# _______________ Building The Linear Regression Model ________________

model = LinearRegression()
model.fit(x, y)


# Printing model score
score = model.score(x, y)
print(score)


# Plotting Linear Line to macth the datapoints with the regression  

plt.scatter(data.Width, y, color="red")
plt.plot(x, model.predict(x), color="black")
plt.title("Predicting Fish Weight from its Width")
plt.xlabel("Width")
plt.ylabel("Weight")
plt.show()

