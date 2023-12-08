import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# The age range corresponding to the errors and predicted values
ages = np.array(range(8, 18))

# The arrays you provided
errors = np.array([8.5, 11.442243589744066, 14.02350815850832, 19.68958333333262, 30.678394522144117, 32.89243589743751, 27.27000000000089, 26.441666666666606, 27.68050990676022, 29.495993589744216])
predicted_values = np.array([1391.775501165501, 1444.5045862470863, 1499.5888461538461, 1561.1824358974359, 1637.6588927738926, 1712.612768065268, 1767.3201923076922, 1797.176625874126, 1814.0968589743586, 1820.294358974359])

# Model the errors as a function of age
error_model = LinearRegression()
error_model.fit(ages.reshape(-1, 1), errors)

# Predict errors for each age
predicted_errors = error_model.predict(ages.reshape(-1, 1))

# Adjust the original predicted values using the predicted errors
adjusted_predicted_values = predicted_values - predicted_errors

# Plot the original and adjusted predicted values
plt.figure(figsize=(10, 5))
plt.scatter(ages, predicted_values, color='blue', label='Original Predictions')
plt.scatter(ages, adjusted_predicted_values, color='red', label='Adjusted Predictions')
plt.plot(ages, predicted_errors, color='green', label='Predicted Errors')
plt.xlabel('Age')
plt.ylabel('Height')
plt.legend()
plt.title('Original vs Adjusted Predictions')
plt.show()

# Function to adjust new predictions using the error model
def adjust_new_predictions(new_ages, new_heights, error_model):
    new_predicted_errors = error_model.predict(new_ages.reshape(-1, 1))
    return new_heights - new_predicted_errors

# Example usage for a new individual
new_individual_ages = np.array([9, 12, 15])  # Example ages for a new individual
new_individual_heights = np.array([1450, 1550, 1650])  # Example heights for a new individual (in cm)
new_individual_adjusted_heights = adjust_new_predictions(new_individual_ages, new_individual_heights, error_model)

print("New Individual Adjusted Heights:", new_individual_adjusted_heights)

new_individual_age_18_prediction = 1820.294358974359
age_18_predicted_error = error_model.predict(np.array([[18]]))  # Predicted error at age 18

# Adjust the prediction for age 18
adjusted_height_at_age_18 = new_individual_age_18_prediction - age_18_predicted_error

print(f"Adjusted Height at Age 18: {adjusted_height_at_age_18[0]} cm")