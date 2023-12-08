import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from statsmodels.tsa.arima.model import ARIMA

os.chdir('src')
# Load your time series data into a Pandas DataFrame
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')
time_series = data['ATV_18']  # Assuming 'ATV_18' is the column you want to forecast

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(time_series)
plt.title('Original Time Series Data')
plt.xlabel('Time')
plt.ylabel('Height at Age 18')
#plt.show()

# Perform the Dickey-Fuller test for stationarity
result = adfuller(time_series)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Critical Values:", result[4])

# Calculate and plot the autocorrelation function (ACF) and partial autocorrelation function (PACF)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(time_series, lags=20, ax=plt.gca(), title='Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(time_series, lags=20, ax=plt.gca(), title='Partial Autocorrelation Function (PACF)')
#plt.show()

# Save the determined values of p, d, and q
p = 1  # Replace with the appropriate value based on PACF plot
d = 1  # Replace with the appropriate value based on stationarity test
q = 1  # Replace with the appropriate value based on ACF plot

print("Selected p, d, q values:", p, d, q)


# Reshape the data to a long format
melted_data = data.melt(id_vars=['child_id'], var_name='age', value_name='height')
melted_data['age'] = melted_data['age'].str.extract('(\d+)').astype(int)

# Sort the data by 'child_id' and 'age'
melted_data = melted_data.sort_values(['child_id', 'age'])

# Train an ARIMA model on the reshaped data
# Specify the appropriate order (p, d, q) based on data analysis
# Use 'height' as the time series and 'age' as the time component
model = ARIMA(melted_data['height'], order=(p, d, q))
model = model.fit()

# Select input data (height at age 8)
input_data = melted_data[melted_data['age'] == 8]['height']

# Make predictions based on input data
predictions = model.forecast(steps=10)  # Forecast height at age 18

# The 'predictions' variable now contains the model's prediction for height at age 18.
print(predictions)
