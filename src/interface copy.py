import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
import os
os.chdir('src')
loaded_model = load_model('my_model.h5')
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

    # Function to save the data and update the Listbox
    def save_data():
        age = float(age_entry.get())
        height = float(height_entry.get())
        input_data.append((age, height))
        data_window.destroy()
        update_listbox()

    save_button = ttk.Button(data_window, text="Save Data", command=save_data)
    save_button.grid(row=2, columnspan=2)

# Function to update the Listbox with entered data
def update_listbox():
    data_listbox.delete(0, tk.END)  # Clear the current content
    for age, height in input_data:
        data_listbox.insert(tk.END, f"Age: {age}, Height: {height} cm")

# GUI function to plot the predicted growth curve
def plot_growth_curve():
    if not input_data:
        print("No data points provided.")
        return

    ages = np.array([age for age, _ in input_data])

    # Predict heights for provided ages using the loaded model
    predicted_heights = loaded_model.predict(ages)

    # Plot the growth curve
    plt.figure(figsize=(10, 5))
    plt.plot(ages, predicted_heights, 'go-', label='Predicted Heights')
    plt.xlabel('Age')
    plt.ylabel('Height (cm)')
    plt.legend()
    plt.grid(True)
    plt.title('Predicted Growth Curve')
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
data_listbox = tk.Listbox(root)
data_listbox.grid(row=2, column=0, columnspan=2)

# Run the main event loop
root.mainloop()