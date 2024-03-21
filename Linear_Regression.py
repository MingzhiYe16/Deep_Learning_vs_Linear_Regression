import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load training data
df_train = pd.read_csv('simtrain5.txt', sep='\s+')
X_train = df_train.iloc[:, 1:].values  # Features: x1 to x13
y_train = df_train.iloc[:, 0].values  # Target: y

# Load test data
df_test = pd.read_csv('simtest5.txt', sep='\s+')
X_test = df_test.iloc[:, 1:].values  # Features: x1 to x13
y_test = df_test.iloc[:, 0].values  # Target: y



# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print MSE and R-squared values
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Mean Squared Error (MSE): 0.0013560355085464643
# R-squared (R²): 0.1951519370213345