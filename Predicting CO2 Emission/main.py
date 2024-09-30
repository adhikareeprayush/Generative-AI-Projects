# Simple Linear Regression
from sklearn.linear_model import LinearRegression  # Correct import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Reading the data
df = pd.read_csv("FuelConsumptionCo2.csv")

# Selecting features to explore
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Plotting histograms to visualize the data
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Train-Test split
msk = np.random.rand(len(df)) < 0.8  # Correct the mask creation
train = cdf[msk]
test = cdf[~msk]

# Plot train data (optional)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# Train the model
regr = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])  # Feature used for training
train_y = np.asanyarray(train[['CO2EMISSIONS']])  # Target

regr.fit(train_x, train_y)  # Train the model

# The coefficients
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)

# Test the model
test_x = np.asanyarray(test[['ENGINESIZE']])  # Feature from the test set
test_y_ = np.asanyarray(test[['CO2EMISSIONS']])  # Ground truth for test set

test_y = regr.predict(test_x)  # Predict the CO2 emissions

predictions = regr.predict(test_x)

# Evaluate the model
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
