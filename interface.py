from scipy import stats
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import os

from main import plot_growth, find_similar_growth_patterns, process_reference_data, find_optimal_top_n, find_best_predicted_height

root = tk.Tk()
root.withdraw()  # Hide the main window

#csv_file_path = filedialog.askopenfilename(title="Select the CSV file",
#                                          filetypes=[("CSV files", "*.csv")])
#if not csv_file_path:
#   raise Exception("A CSV file must be selected to proceed.")

#Load your data
#data = pd.read_csv(csv_file_path)
data= pd.read_csv('svk_height_weight_mens_2008_v2.csv')
#print(data)
interpolated_growth_data = process_reference_data(data)
interpolated_growth_data = interpolated_growth_data.drop(columns=['child_id'])

class GrowthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Growth Prediction")
        
        #Placeholder for iqr and predicted age at 18
        self.predicted_height_at_18 = None
        self.iqr = None
        # Age input
        self.age_label = tk.Label(self, text="Age Height (separate multiple measurements by commas):")
        self.age_label.pack()
        self.age_entry = tk.Entry(self)
        self.age_entry.pack()

       
        # Button to add data to the list
        self.add_button = tk.Button(self, text="1. Add Data", command=self.add_data)
        self.add_button.pack()

        # Display the data
        self.data_listbox = tk.Listbox(self)
        self.data_listbox.pack()
        # Button to plot the graph
        self.plot_button = tk.Button(self, text="2. Plot Growth Curve", command=self.plot_growth_curve)
        self.plot_button.pack()
        self.canvas = None

        #display range of heights

        self.best_height_button = tk.Button(self, text="Find Best Predicted Height", command=self.display_best_predicted_height)
        self.best_height_button.pack()

        self.results_text_widget = tk.Text(self, height=10, width=50)
        self.results_text_widget.pack()
        # Actual height at 18 input
        self.actual_height_label = tk.Label(self, text="Actual Height at 18:")
        self.actual_height_label.pack()
        self.actual_height_entry = tk.Entry(self)
        self.actual_height_entry.pack()
        # Button to find optimal top_n
        self.find_optimal_top_n_button = tk.Button(self, text="Find Optimal N of Curves", command=self.find_optimal_top_n)
        self.find_optimal_top_n_button.pack()

        # Display for optimal top_n and predicted height
        self.optimal_top_n = 100
        self.optimal_top_n_label = tk.Label(self, text="")
        self.optimal_top_n_label.pack()

        # List to store (age, height) pairs
        self.age_height_pairs = []

        
        # Clear the data
        self.clear_button = tk.Button(self, text="Clear Data", command=self.clear_data)
        self.clear_button.pack()




    def display_best_predicted_height(self):
        if self.predicted_height_at_18 is None or self.iqr is None:
            messagebox.showerror("Error", "Predicted height at 18 and IQR not available.")
            return

        min_height = self.predicted_height_at_18 - self.iqr
        max_height = self.predicted_height_at_18 + self.iqr

        predicted_heights_range = np.arange(min_height, max_height, 0.5)
        print(predicted_heights_range)
        optimal_height, results = find_best_predicted_height(self.age_height_pairs, interpolated_growth_data, predicted_heights_range)

        #9 139.4, 10 144.9

        # Display the results
        results_text = "Predicted Height at 18 and Corresponding Top N:\n"
        results_text += "\n".join([f"Height: {height:.1f} cm, Top N: {top_n}" for height, top_n in results.items()])
        results_text += f"\n\nOptimal Predicted Height at 18: {optimal_height:.1f} cm"
        # Display the results in the text widget
        self.results_text_widget.delete('1.0', tk.END)  # Clear existing text
        self.results_text_widget.insert(tk.END, results_text)


    def add_data(self):
        try:
            entries = self.age_entry.get().split(',')
            for entry in entries:
                age, height = map(float, entry.strip().split())
                
                if not 8 <= age <= 18:
                    messagebox.showerror("Error", "Age should be between 8 and 18.")
                    return
                
                # Check if the height is realistic and greater than any younger age's height
                if not 50 <= height <= 250:
                    messagebox.showerror("Error", "Height should be a realistic value.")
                    return
                
                # Check that the height is not less than any height for a younger age
                for existing_age, existing_height in self.age_height_pairs:
                    if age > existing_age and height < existing_height:
                        messagebox.showerror("Error", "Height for each age must not be less than the height for any younger age.")
                        return
                    if age < existing_age and height > existing_height:
                        messagebox.showerror("Error", "Height for each age must be greater than the height for any older age.")
                        return
                
                # If validation passes, add the data
                self.age_height_pairs.append((age, height))
            
            self.age_height_pairs.sort()  # Keep the list sorted by age
            self.update_data_listbox()
            self.age_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Invalid input for age or height.")

    def find_optimal_top_n(self):
        actual_height_at_18 = self.actual_height_entry.get().strip()

        if actual_height_at_18:  # Check if the actual height at 18 is entered
            try:
                actual_height_at_18 = float(actual_height_at_18)
                top_n_range = range(10, 500)  # Adjust the range as needed

                optimal_top_n, optimal_predicted_height = find_optimal_top_n(self.age_height_pairs, actual_height_at_18, interpolated_growth_data, top_n_range)
                self.optimal_top_n = optimal_top_n  # Save the optimal top_n
                self.optimal_top_n_label.config(text=f"Optimal Top N: {optimal_top_n}\nPredicted Height at 18: {optimal_predicted_height:.2f} cm")
            
            except ValueError:
                messagebox.showerror("Error", "Invalid input for actual height at 18.")
        else:
            # If actual height at 18 is not entered, use the default top_n value
            self.optimal_top_n = 100
            self.optimal_top_n_label.config(text="Using default Top N: 100")
        print(optimal_top_n)


    def update_data_listbox(self):
        self.data_listbox.delete(0, tk.END)
        for age, height in self.age_height_pairs:
            self.data_listbox.insert(tk.END, f"Age {age}, Height {height} cm")
        
    def clear_data(self):
        self.age_height_pairs.clear()
        self.data_listbox.delete(0, tk.END)
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def plot_growth_curve(self):
        if not self.age_height_pairs:
            messagebox.showerror("Error", "No data to plot.")
            return
        
        fig, self.predicted_height_at_18, self.iqr = plot_growth(self.age_height_pairs, interpolated_growth_data, self.optimal_top_n)
        self.display_plot(fig)


    def display_plot(self, fig):
        # If a previous figure exists, clear it before plotting a new one
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create a new canvas and add it to the GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # # Display data table
        # table_frame = tk.Frame(self)
        # table_frame.pack(side=tk.TOP, pady=10)
        # data_table_widget = tk.Text(table_frame, wrap=tk.WORD, height=5, width=30)
        # data_table_widget.insert(tk.END, data_table)
        # data_table_widget.config(state=tk.DISABLED)
        # data_table_widget.pack()


if __name__ == "__main__":
    app = GrowthApp()
    app.mainloop()