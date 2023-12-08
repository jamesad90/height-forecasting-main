import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.h5')
# Provided data
ages = np.array(range(8, 19))
errors = np.array([6.5, 8.553141025641935, 11.442243589744066, 14.02350815850832, 19.68958333333262, 30.678394522144117, 32.89243589743751, 27.27000000000089, 26.441666666666606, 27.68050990676022, 29.495993589744216])
predicted_values = np.array([1335, 1391.775501165501, 1444.5045862470863, 1499.5888461538461, 1561.1824358974359, 1637.6588927738926, 1712.612768065268, 1767.3201923076922, 1797.176625874126, 1814.0968589743586, 1820.294358974359])


errors = errors /10
predicted_values = predicted_values / 10
print(predicted_values)
# Model the errors as a function of age
error_model = LinearRegression()
error_model.fit(ages.reshape(-1, 1), errors)

# Lists to store the input data
input_data = []

# Create a function to add data
def add_data():
    # Create a new window to input age and height
    data_window = tk.Toplevel(root)
    data_window.title("Add Data")
    
    age_label = ttk.Label(data_window, text="Age:")
    age_label.grid(row=0, column=0)
    height_label = ttk.Label(data_window, text="Height (cm):")
    height_label.grid(row=1, column=0)
    
    age_entry = ttk.Entry(data_window)
    age_entry.grid(row=0, column=1)
    height_entry = ttk.Entry(data_window)
    height_entry.grid(row=1, column=1)
    
    # Function to save the data and close the window
    def save_data():
        age = float(age_entry.get())
        height = float(height_entry.get())
        input_data.append((age, height))
        data_window.destroy()
    
    save_button = ttk.Button(data_window, text="Save Data", command=save_data)
    save_button.grid(row=2, columnspan=2)

# GUI function to plot the predicted growth curve
def plot_growth_curve():
    if not input_data:
        print("No data points provided.")
        return

    ages = np.array([age for age, _ in input_data])
    heights = np.array([height for _, height in input_data])

    # Model the errors as a function of age
    error_model = LinearRegression()
    error_model.fit(ages.reshape(-1, 1), heights - predicted_values)

    # Predict the error at age 18 using the error model
    age_18_predicted_error = error_model.predict(np.array([[18]]))

    # Adjust the predicted height at age 18
    adjusted_height_at_18 = predicted_values + age_18_predicted_error[0]

    # Plot the growth curve
    plt.figure(figsize=(10, 5))
    plt.plot(ages, heights, 'bo', label='Input Data')
    plt.plot(ages, predicted_values, 'go-', label='Predicted Heights')
    plt.plot(18, adjusted_height_at_18, 'ro', label='Adjusted Height at 18')
    plt.xlabel('Age')
    plt.ylabel('Height (cm)')
    plt.legend()
    plt.grid(True)
    plt.title('Predicted Growth Curve with Adjusted Height at Age 18')
    plt.show()

# Create the main application window
root = tk.Tk()
root.title("Growth Curve Predictor")

# Create a button to add data
add_data_button = ttk.Button(root, text="Add Data", command=add_data)
add_data_button.grid(row=0, column=0)

# Create a button to plot the growth curve
plot_button = ttk.Button(root, text="Predict and Plot Growth Curve", command=plot_growth_curve)
plot_button.grid(row=1, column=0)

# Run the main event loop
root.mainloop()