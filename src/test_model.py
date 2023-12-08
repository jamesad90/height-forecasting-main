import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import joblib
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
os.chdir('src')

loaded_model = load_model('my_model.keras')
def test_model_with_pairs(model, age_height_pairs):
    # Extract ages and heights
    ages = np.array([pair[0] for pair in age_height_pairs]).reshape(-1, 1)
    heights = np.array([pair[1] for pair in age_height_pairs])
    

    
    # Predict heights using the model
    predicted_heights = model.predict(ages).flatten()
    
    # Print the results
    for i, (age, actual_height) in enumerate(age_height_pairs):
        print(f"Age: {age}, Actual Height: {actual_height} cm, Predicted Height at 18: {predicted_heights[i]} cm")

# Assume model and scaler are already defined and the model is trained
# Example list of age-height pairs to test the model
age_height_pairs_to_test = [(8, 130), (10, 140), (12, 150), (14, 160), (16, 170)]

# Testing the model with the provided list of age-height pairs
test_model_with_pairs(loaded_model, age_height_pairs_to_test)













model = joblib.load("random_forest.joblib")
#curve features model
def calculate_parameters(age_data, height_data):
    # Define a function to fit a growth curve to the data
    def growth_curve(age, a, b, c):
        # Here you can define the form of the growth curve, such as a logistic curve
        return a / (1 + np.exp(b - c * age))

    # Fit the growth curve
    try:
        params, _ = curve_fit(growth_curve, age_data, height_data, maxfev=5000)
    except RuntimeError as e:
        print(f"Could not fit curve: {e}")
        return None

    # Fit a spline to estimate the first and second derivatives
    spline = UnivariateSpline(age_data, height_data, k=3, s=0)
    first_derivative = spline.derivative(n=1)
    second_derivative = spline.derivative(n=2)

    # Estimate peak height velocity (PHV)
    velocities = first_derivative(age_data)
    phv_age = age_data[np.argmax(velocities)]
    phv_value = np.max(velocities)

    # Calculate acceleration at PHV age
    acceleration = second_derivative(phv_age)
    print(phv_age)
    return {
        'PHV_Age': phv_age,
        'PHV_Value': phv_value,
        'Acceleration': acceleration,
        'Curve_Param_a': params[0],  # Assuming 'params' is an array of curve parameters
        'Curve_Param_b': params[1],
        'Curve_Param_c': params[2]
    }
    

def predict_height_with_rf(user_input):
    # Load the trained Random Forest model
    rf_model = load("random_forest.joblib")

    # List to store predicted heights for each set of user input
    predicted_heights = []

    for input_set in user_input:
        age_data, height_data = input_set  # Replace with how the user provides input

        # Calculate parameters for the current input set
        params = calculate_parameters(age_data, height_data)

        if params is not None:
            # Create a DataFrame from the calculated parameters
            user_data = pd.DataFrame([params])

            # Make a prediction using the trained model
            predicted_height = rf_model.predict(user_data)
            predicted_heights.append(predicted_height[0])
            print(predicted_heights)
        else:
            predicted_heights.append(None)

    return predicted_heights

# Example usage:
user_input = [([(8, 10, 11, 12, 13), (130, 135, 140, 150, 160)])]

# Predict final height for user input
predicted_final_heights = []
for ages, heights in user_input:
    params = calculate_parameters(ages, heights)
    if params is not None:
        final_height = params[7]  # You can adjust this to extract the desired parameter
        predicted_final_heights.append(final_height)
import matplotlib.pyplot as plt
# Plotting the predicted final heights
plt.figure(figsize=(10, 6))

for ages, heights in user_input:
    plt.plot(ages, heights, 'o-', label='User Input')

plt.xlabel('Age')
plt.ylabel('Height (cm)')
plt.title('Predicted Final Height')
plt.legend()
plt.grid(True)
plt.show()














#nnetwork model
#loaded_model = load_model('my_model.keras')
scaler = load('scaler.joblib')


