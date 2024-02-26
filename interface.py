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

from main import plot_growth, find_similar_growth_patterns, process_reference_data, find_best_predicted_height

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
        self.method = 'euclidean' #Default Method
        # Placeholder for iqr and predicted age at 18
        self.predicted_height_at_18 = None
        self.iqr = None

        # Age input
        self.age_label = tk.Label(self, text="Age (separate multiple measurements by new lines):")
        self.age_label.pack()
        self.age_entry = tk.Text(self, height=5, width=10)
        self.age_entry.pack()

        # Height input
        self.height_label = tk.Label(self, text="Height (separate multiple measurements by new lines):")
        self.height_label.pack()
        self.height_entry = tk.Text(self, height=5, width=10)
        self.height_entry.pack()

        # Button to add data
        self.add_button = tk.Button(self, text="1. Add Data", command=self.add_data)
        self.add_button.pack()

        # Display the data
        self.data_listbox = tk.Listbox(self)
        self.data_listbox.pack()


        # Label to display current method
        self.method_label = tk.Label(self, text=f"Current Method: {self.method}")
        self.method_label.pack()
        # Button to toggle method
        self.toggle_method_button = tk.Button(self, text="Toggle Method", command=self.toggle_method)
        self.toggle_method_button.pack()



        # Button to plot the graph
        self.plot_button = tk.Button(self, text="2. Plot Growth Curve", command=self.plot_growth_curve)
        self.plot_button.pack()
        self.canvas = None

        # # Display range of heights
        # self.best_height_button = tk.Button(self, text="Find Best Predicted Height", command=self.display_best_predicted_height)
        # self.best_height_button.pack()

        # self.results_text_widget = tk.Text(self, height=5, width=15)
        # self.results_text_widget.pack()

        # # Actual height at 18 input
        # self.actual_height_label = tk.Label(self, text="Actual Height at 18:")
        # self.actual_height_label.pack()
        # self.actual_height_entry = tk.Entry(self)
        # self.actual_height_entry.pack()

        # # Button to find optimal top_n
        # self.find_optimal_top_n_button = tk.Button(self, text="Find Optimal N of Curves", command=self.find_optimal_top_n)
        # self.find_optimal_top_n_button.pack()

        # Display for optimal top_n and predicted height
        # self.optimal_top_n = 100
        # self.optimal_top_n_label = tk.Label(self, text="")
        # self.optimal_top_n_label.pack()

        # List to store (age, height) pairs
        self.age_height_pairs = []

        # Clear the data
        self.clear_button = tk.Button(self, text="Reset", command=self.clear_data)
        self.clear_button.pack()

    def toggle_method(self):
        # Toggle between methods
        if self.method == 'euclidean':
            self.method = 'cosine'
        else:
            self.method = 'euclidean'

        # Update the method label to reflect the current method
        self.method_label.config(text=f"Current Method: {self.method}")
        

    # def display_best_predicted_height(self):
    #     if self.predicted_height_at_18 is None or self.iqr is None:
    #         messagebox.showerror("Error", "Predicted height at 18 and IQR not available.")
    #         return

    #     min_height = self.predicted_height_at_18 - self.iqr
    #     max_height = self.predicted_height_at_18 + self.iqr

    #     predicted_heights_range = np.arange(min_height, max_height, 0.5)
    #     print(predicted_heights_range)
    #     optimal_height, results = find_best_predicted_height(self.age_height_pairs, interpolated_growth_data, predicted_heights_range)

    #     #9 139.4, 10 144.9

    #     # Display the results
    #     results_text = "Predicted Height at 18 and Corresponding Top N:\n"
    #     results_text += "\n".join([f"Height: {height:.1f} cm, Top N: {top_n}" for height, top_n in results.items()])
    #     results_text += f"\n\nOptimal Predicted Height at 18: {optimal_height:.1f} cm"
    #     # Display the results in the text widget
    #     self.results_text_widget.delete('1.0', tk.END)  # Clear existing text
    #     self.results_text_widget.insert(tk.END, results_text)


    def add_data(self):
        try:
            # Get the pasted data for age and height
            age_data = self.age_entry.get("1.0", tk.END).strip()
            height_data = self.height_entry.get("1.0", tk.END).strip()

            # Split the pasted data into lists of ages and heights
            age_list = age_data.split('\n')
            height_list = height_data.split('\n')

            # Validate that the number of ages matches the number of heights
            if len(age_list) != len(height_list):
                messagebox.showerror("Error", "Number of ages must match number of heights.")
                return

            # Process each pair of age and height
            for age, height in zip(age_list, height_list):
                age = float(age.strip())
                height = float(height.strip())
                
                # Validate age and height
                if not 8 <= age <= 18:
                    messagebox.showerror("Error", "Age should be between 8 and 18.")
                    return
                if not 50 <= height <= 250:
                    messagebox.showerror("Error", "Height should be a realistic value.")
                    return

                # Add the validated data to the list
                self.age_height_pairs.append((age, height))
            
            # Sort the list by age
            self.age_height_pairs.sort()

            # Update the data listbox
            self.update_data_listbox()

            # Clear the text entry widgets
            self.age_entry.delete("1.0", tk.END)
            self.height_entry.delete("1.0", tk.END)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    # def find_optimal_top_n(self):
    #     actual_height_at_18 = self.actual_height_entry.get().strip()

    #     if actual_height_at_18:  # Check if the actual height at 18 is entered
    #         try:
    #             actual_height_at_18 = float(actual_height_at_18)
    #             top_n_range = range(10, 500)  # Adjust the range as needed

    #             optimal_top_n, optimal_predicted_height = find_optimal_top_n(self.age_height_pairs, actual_height_at_18, interpolated_growth_data, top_n_range)
    #             self.optimal_top_n = optimal_top_n  # Save the optimal top_n
    #             self.optimal_top_n_label.config(text=f"Optimal Top N: {optimal_top_n}\nPredicted Height at 18: {optimal_predicted_height:.2f} cm")
            
    #         except ValueError:
    #             messagebox.showerror("Error", "Invalid input for actual height at 18.")
    #     else:
    #         # If actual height at 18 is not entered, use the default top_n value
    #         self.optimal_top_n = 100
    #         self.optimal_top_n_label.config(text="Using default Top N: 100")
    #     print(optimal_top_n)


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
        
        fig, self.predicted_height_at_18, self.iqr = plot_growth(self.age_height_pairs, interpolated_growth_data, 100, self.method)
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