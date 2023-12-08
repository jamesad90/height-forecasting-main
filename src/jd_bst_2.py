import os
os.chdir('src')
def calculate_aphv(row):
    height_measurements = [row['ATV_8'], row['ATV_9'], row['ATV_10'], row['ATV_11'], row['ATV_12'], row['ATV_13'], row['ATV_14'], row['ATV_15'], row['ATV_16'], row['ATV_17'], row['ATV_18']]
    age_points = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    age_at_aphv = age_points[np.argmax(np.diff(height_measurements))]
    return age_at_aphv

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load the data from the CSV file
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Calculate APHV for each participant
#data['APHV'] = data.apply(calculate_aphv, axis=1)

# Train the model on the full data
X = data.drop(['child_id'], axis=1)  # All columns except 'child_id' and the target variable
X = X.loc[:, ~X.columns.str.contains('ATT_')]
y = data['ATV_18']  # Target variable: height at age 18
print(X)

# Train the model on the full data
X = data.drop(['child_id', 'ATV_18'], axis=1)  # All columns except 'child_id' and the target variable
X = X.loc[:, ~X.columns.str.contains('ATT_')]  # Drop columns with 'ATT_' in their names
y = data['ATV_18']  # Target variable: height at age 18

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
# Function to assess the model
def assess_model(model, poly, X_poly, y):
    # Make predictions
    y_pred = model.predict(X_poly)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return mae, mse


# Prediction function for new individual data
def predict_heights(model, poly, age_height_pairs):
    predicted_heights = []
    for age, height in age_height_pairs:
        input_data = np.zeros((1, 11))
        input_data[0, age - 8] = height
        #input_data[0, -1] = historical_aphv
        input_data_poly = poly.transform(input_data)
        predicted_height = model.predict(input_data_poly)[0]
        predicted_heights.append(predicted_height)
    return predicted_heights


# Plot the actual and predicted growth trajectories
def plot_growth_trajectories(data, age_height_pairs, predicted_heights):
    age_columns = ['ATV_8', 'ATV_9', 'ATV_10', 'ATV_11', 'ATV_12', 'ATV_13', 'ATV_14', 'ATV_15', 'ATV_16', 'ATV_17', 'ATV_18']
    ages = np.array(range(8, 19))
    
    for index, row in data.iterrows():
        if row[age_columns].isnull().any():
            continue  # Skip this row if there are any null values in the age columns
        plt.plot(ages, row[age_columns], color='gray', lw=0.5)
    
    # Plot the input age and height pairs
    for age, height in age_height_pairs:
        plt.scatter(age, height, color='red', label='Input Data Point')
    
    # Plot the predicted heights at age 18
    for age, predicted_height in zip([pair[0] for pair in age_height_pairs], predicted_heights):
        plt.scatter(18, predicted_height, label=f'Predicted Height at 18 for Age {age}')
    
    plt.xlabel('Age')
    plt.ylabel('Height (cm)')
    plt.title('Actual and Predicted Growth Trajectories')
    plt.legend()
    plt.show()

# Plot the growth trajectories with input data and predicted heights

# Example usage with a new individual
age_height_pairs = [(10, 1100), (13, 1300), (15, 1500)]
#historical_aphv = 14  # This should be the APHV calculated from historical data
predicted_height = predict_heights(model, poly, age_height_pairs)
print(f"Predicted height at age 18: {predicted_height[0]}")
# Plot the growth trajectories
plot_growth_trajectories(data, age_height_pairs, predicted_height)
# Assess the model
mae, mse = assess_model(model, X_poly, y)

# Print the predicted heights and evaluation metrics
for age, height, predicted_height in zip(age_height_pairs, predicted_height):
    print(f"Predicted height at age 18 for age {age} and height {height}: {predicted_height}")
    
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")