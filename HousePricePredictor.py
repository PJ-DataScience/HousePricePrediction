# Importing libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read data from CSV file

data = pd.read_csv("./data.csv")

# Transform Ordinal data to numerical data

data_encoder = OrdinalEncoder()
data_encoder.fit(np.array(data.iloc[:, 2]).reshape(-1, 1))
data['type_of_building'] = data_encoder.transform(np.array(data.iloc[:, 2]).reshape(-1, 1))

# Independent and Dependent variable separation

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Divide the dataset into training and test datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create model instance

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the values for test data

predicted_price = regressor.predict([[3,5,0,15]])

# Print the results

print("The result is: ", predicted_price)

# The predicted result is

## The result is:  [3032964.6958741]
