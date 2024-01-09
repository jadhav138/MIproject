# importing pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Providing the absolute path to the file
file_path = r'C:\Users\HP\Desktop\Big data Final Project\genetic_markers_dataset.csv'

# Reading the CSV file using the absolute path
dataset = pd.read_csv(file_path)

# Checking the dataset
print(dataset)
print(dataset.columns)
print(dataset.head())

# Separating features (X) and target variable (y)
X = dataset.drop('Disease_Likelihood', axis=1)  # Features
y = dataset['Disease_Likelihood']  # Target variable
print(X)
print(y)

# Splitting the data into 70% training and 30% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor()

# Training the model on the training data
model.fit(X_train, y_train)

# Predicting the target values on the validation set
predictions = model.predict(X_val)

# Evaluate the model (e.g., using accuracy, RMSE, etc., depending on the problem)
# For regression, you might use metrics like mean squared error (MSE)
mse = mean_squared_error(y_val, predictions)
print(f"Mean Squared Error on Validation Set: {mse}")

# Calculating Mean Squared Error
def calculate_mse(y_val, predictions):
    mse = mean_squared_error(y_val, predictions)
    return mse

mse_value = calculate_mse(y_val, predictions)
print(f"Mean Squared Error on Validation Set: {mse_value}")

# Plotting the MSE
plt.figure(figsize=(8, 6))
plt.bar("Random Forest", mse, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance - Mean Squared Error')
plt.show()
