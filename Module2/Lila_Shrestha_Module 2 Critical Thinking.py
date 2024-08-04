#import libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = 'https://gist.githubusercontent.com/gurchetan1000/ec90a0a8004927e57c24b20a6f8c8d35/raw/fcd83b35021a4c1d7f1f1d5dc83c07c8ffc0d3e2/iris.csv'
data = pd.read_csv(url)

# Display the first few rows and column names to verify
print(data.head())
print(data.columns)

# Use the correct target column 'Name'
X = data.drop('Name', axis=1).values
y = data['Name'].values

# Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the value of k
k = 3

# calculation of Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# predicting the class of a single test instance
def predict(X_train, y_train, x_test, k):
    # Calculate the distance between the test instance and all training instances
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    
    # Get the indices of the k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    
    # Get the classes of the k nearest neighbors
    k_nearest_labels = [y_train[i] for i in k_indices]
    
    # Return the most common class among the k nearest neighbors
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Function to predict the classes of all test instances
def predict_all(X_train, y_train, X_test, k):
    return [predict(X_train, y_train, x_test, k) for x_test in X_test]

# Predict the classes for the test data
y_pred = predict_all(X_train, y_train, X_test, k)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Printing the classification report
print(classification_report(y_test, y_pred))

# For new instance that accept user input
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Create a new instance with the input values
new_instance = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Predict the class for the new instance
predicted_class = predict(X_train, y_train, new_instance, k)
print(f'The predicted class for the input: {predicted_class}')
