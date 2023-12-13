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

from main import plot_growth, find_similar_growth_patterns, process_reference_data

root = tk.Tk()
root.withdraw()  # Hide the main window

csv_file_path = filedialog.askopenfilename(title="Select the CSV file",
                                          filetypes=[("CSV files", "*.csv")])
if not csv_file_path:
   raise Exception("A CSV file must be selected to proceed.")

#Load your data
data = pd.read_csv(csv_file_path)
print(data)
interpolated_growth_data = process_reference_data(data)
interpolated_growth_data = interpolated_growth_data.drop(columns=['child_id'])

class GrowthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Growth Prediction")

        # List to store (age, height) pairs
        self.age_height_pairs = []

        # Age input
        self.age_label = tk.Label(self, text="Age Height (separate multiple measurements by commas):")
        self.age_label.pack()
        self.age_entry = tk.Entry(self)
        self.age_entry.pack()

       
        # Button to add data to the list
        self.add_button = tk.Button(self, text="Add Data", command=self.add_data)
        self.add_button.pack()

        # Display the data
        self.data_listbox = tk.Listbox(self)
        self.data_listbox.pack()
        
        # Clear the data
        self.clear_button = tk.Button(self, text="Clear Data", command=self.clear_data)
        self.clear_button.pack()

        # Button to plot the graph
        self.plot_button = tk.Button(self, text="Plot Growth Curve", command=self.plot_growth_curve)
        self.plot_button.pack()
        self.canvas = None

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
        
        fig = plot_growth(self.age_height_pairs, interpolated_growth_data)
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