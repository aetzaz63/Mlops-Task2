import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = pd.read_csv('salary_data.csv')

# Define features and target
X = data['YearsExperience'].values.reshape(-1, 1)  # Independent variable (Years of Experience)
y = data['Salary'].values  # Dependent variable (Salary)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
