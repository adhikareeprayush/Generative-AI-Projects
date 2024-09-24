# Building a Decision Tree to Predict Customer Churn

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np

warnings.filterwarnings('ignore')

# Creating a synthetic dataset
data = {
    'CustomerID': range(1, 101),
    'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 10,
    'MonthlyCharge': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140] * 10,
    'CustomerServiceCalls': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0] * 10,
    'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'] * 10
}

df = pd.DataFrame(data)

# Converting 'Churn' column to numerical values
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Splitting the dataset into features and target variable
X = df[['Age', 'MonthlyCharge', 'CustomerServiceCalls']]
y = df['Churn']

# 70% of the data is used for training and 30% is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=4)  # Optional: Limiting depth to avoid overfitting
clf.fit(X_train, y_train)

# Making Predictions on the test set
y_pred = clf.predict(X_test)

# Evaluating the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Dynamically setting the class names based on unique values in y_train
class_names = [str(cls) for cls in np.unique(y_train)]

# Visualizing the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=['Age', 'MonthlyCharge', 'CustomerServiceCalls'],
               class_names=class_names)  # Dynamically passed class names
plt.show()
