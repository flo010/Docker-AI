import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import joblib  # For saving and loading the model

# Example Dataset: Random dataset of unmber of rooms with pricing
X = np.array([[1], [1], [2], [2], [3], [3], [3], [3], [4], [4]])  # Number of rooms
y = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 600])   # Housing prices in $1000

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
model_filename = ""
with open('/app/models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)


