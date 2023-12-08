import os
os.chdir('src')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
# Define the Preece-Baines Model I function
def preece_baines_model_i(age, a, b, c, d, g):
    return a + (b - a) / ((1 + np.exp(c - (age / d))) ** g)

# Load the data from the CSV file
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')
height_columns = [col for col in data.columns if col.startswith('ATV_')]
data[height_columns] = data[height_columns] / 10

# Calculate Z-scores for each height column and filter out outliers
z_scores = stats.zscore(data[height_columns])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]
# Prepare your data for fitting
# Create a long-form dataframe where each row corresponds to a single age-height observation
melted_data = data.melt(id_vars='child_id', value_vars=[f'ATV_{i}' for i in range(8, 19)],
                        var_name='Age', value_name='Height')

# Extract the ages and convert them to numeric values
melted_data['Age'] = melted_data['Age'].str.extract('(\d+)').astype(int)

# Drop rows with missing height values
melted_data = melted_data.dropna(subset=['Height'])
print(melted_data)

# Fit the model to your data
ages = melted_data['Age'].values
heights = melted_data['Height'].values

initial_guess = [heights.min(), heights.max(), 0.5, 10, 1]  # Modify based on your data


params, _ = curve_fit(preece_baines_model_i, ages, heights, p0=initial_guess)


ages_to_predict = np.array([6, 7, 8,18])  # Replace with the ages you want to predict
predicted_heights = preece_baines_model_i(ages_to_predict, *params)

# Visualize the fitted curve and extrapolated predictions
ages_fit = np.linspace(ages.min(), ages.max(), 100)
heights_fit = preece_baines_model_i(ages_fit, *params)

plt.scatter(ages, heights, label='Data')
plt.plot(ages_fit, heights_fit, color='red', label='Preece-Baines Model I Fit')
plt.scatter(ages_to_predict, predicted_heights, color='blue', label='Predicted Heights')
plt.xlabel('Age')
plt.ylabel('Height')
plt.legend()
plt.show()


for age, height in zip(ages_to_predict, predicted_heights):
    print(f"Predicted height at age {age}: {height}")



# Provide bounds for the parameters to help guide the fitting process
param_bounds = ([85, 100, 5, 10, 0.1],  # Lower bounds for each parameter
                [200, 250, 15, 20, 3])    # Upper bounds for each parameter

# Specify no bounds (or wide enough bounds)
params, covariance = curve_fit(preece_baines_model_i, ages, heights, p0=initial_guess, maxfev=10000)






# Extract the parameters
a, b, c, d, g = params


# Use the fitted parameters to plot the model against the data
ages_fit = np.linspace(ages.min(), ages.max(), 100)
heights_fit = preece_baines_model_i(ages_fit, a, b, c, d, g)
print(params)
plt.scatter(ages, heights, label='Data')
plt.plot(ages_fit, heights_fit, color='red', label='Preece-Baines Model I Fit')
plt.xlabel('Age')
plt.ylabel('Height')
plt.legend()
plt.show()

# Predict the height at age 18 using the fitted model
predicted_height_at_18 = preece_baines_model_i(18, a, b, c, d, g)
print(f"The predicted height at age 18 is: {predicted_height_at_18}")


# Limited height data for the individual
individual_ages = np.array([12, 15, 16])  # Ages at which you have height measurements
individual_heights = np.array([1400, 1500, 1600])  # Corresponding heights

population_params = params
# Predict the height at age 18 for the individual using the population model
predicted_height_at_18_i = preece_baines_model_i(18, *population_params)

# Print the predicted height
print(f"Predicted height at age 18 for the individual: {predicted_height_at_18_i}")