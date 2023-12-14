import os
os.chdir('src')
from scipy import stats
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Load your data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Convert height measurements from mm to cm
height_columns = [col for col in data.columns if col.startswith('ATV_')]
data[height_columns] = data[height_columns] / 10

# Melt the DataFrame
melted_data = pd.melt(data, id_vars=['child_id'], value_vars=height_columns, var_name='Age', value_name='Height')

# Convert age to integer
melted_data['Age'] = melted_data['Age'].str.extract('(\d+)').astype(int)

# Filter out outliers using z-scores
z_scores = stats.zscore(melted_data['Height'])
melted_data = melted_data[(np.abs(z_scores) < 3)]

# Group data by child_id
grouped = melted_data.groupby('child_id')
# Define a function to fit a growth curve to the data
def growth_curve(age, a, b, c):
    # Here you can define the form of the growth curve, such as a logistic curve
    return a / (1 + np.exp(b - c * age))

# Dictionary to hold the extracted features for each child
features_dict = {}

# Loop through each child's data
for child_id, group_data in grouped:
    age_data = group_data['Age'].values
    height_data = group_data['Height'].values

    # Skip the iteration if there's not enough data
    if len(age_data) < 3:
        continue

    # Fit the growth curve
    try:
        params, _ = curve_fit(growth_curve, age_data, height_data, maxfev=5000)
    except RuntimeError as e:
        print(f"Could not fit curve for child_id {child_id}: {e}")
        continue

    # Fit a spline to estimate the first and second derivatives
    spline = UnivariateSpline(age_data, height_data, k=4, s=0)
    first_derivative = spline.derivative(n=1)
    second_derivative = spline.derivative(n=2)

    # Estimate peak height velocity (PHV)
    velocities = first_derivative(age_data)
    phv_age = age_data[np.argmax(velocities)]
    phv_value = np.max(velocities)
    final_height = np.max(height_data)

    # Calculate acceleration at PHV age
    acceleration = second_derivative(phv_age)

    # Store the extracted features in the dictionary
    features_dict[child_id] = {
        'PHV_Age': phv_age,
        'PHV_Value': phv_value,
        'Acceleration': acceleration,
        'Curve_Param_a': params[0],  # Assuming 'params' is an array of curve parameters
        'Curve_Param_b': params[1],
        'Curve_Param_c': params[2],
        'Final_Height': final_height,
        'Max_Height_Predicted': growth_curve(18, *params)  # Predict max height at age 18
    }


# Now 'features_dict' contains the extracted features for each child
# Convert the features_dict into a DataFrame
features_df = pd.DataFrame.from_dict(features_dict, orient='index')
print(features_df)


from sklearn.model_selection import train_test_split

X = features_df.drop('Final_Height', axis=1)  # Features
y = features_df['Final_Height']  
print(y)             # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print(X_train, y_train)
# Train the model
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error

# Make predictions on the test set
y_pred = model.predict(X_test)

print(y_pred)
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')
import joblib
joblib.dump(model, "random_forest.joblib")
