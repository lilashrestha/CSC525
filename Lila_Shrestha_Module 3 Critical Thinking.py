# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
data = pd.read_csv(r'D:\CSU Global\MS in Data Analytics\CSC525\Module3\Salary_Data.csv')

# Explore the data
print(data.head())
print(data.describe())
print(data.isnull().sum())

# Preprocess the data
X = data[['YearsExperience']].values
y = data['Salary'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the polynomial regression model
degree = 3  # You can change this value to try different polynomial degrees
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_poly_train, y_train)

# Make predictions
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'Train R^2: {train_r2}')
print(f'Test R^2: {test_r2}')

# Predict the salary for a given number of years of experience
def predict_salary(years_experience):
    years_experience = np.array([[years_experience]])
    years_experience_poly = poly_features.transform(years_experience)
    predicted_salary = model.predict(years_experience_poly)
    return predicted_salary[0]

# Input from user
years_of_experience = float(input("Enter the years of experience: "))
predicted_salary = predict_salary(years_of_experience)
print(f'The predicted salary for {years_of_experience} years of experience is: ${predicted_salary:.2f}')
# Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(poly_features.transform(X_range))
plt.plot(X_range, y_range_pred, color='Maroon', label='Polynomial Regression Model')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
