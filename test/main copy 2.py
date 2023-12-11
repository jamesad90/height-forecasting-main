from scipy import stats
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import os
from scipy.interpolate import interp1d


root = tk.Tk()
root.withdraw()  # Hide the main window

csv_file_path = filedialog.askopenfilename(title="Select the CSV file",
                                           filetypes=[("CSV files", "*.csv")])
if not csv_file_path:
    raise Exception("A CSV file must be selected to proceed.")

# Load your data
# Load your data
growth_data = pd.read_csv(csv_file_path)

# Remove 'child_id' for outlier detection and processing
#growth_data = data.drop(columns=['child_id'])
atv_columns = [col for col in growth_data.columns if 'ATV_' in col]

# Filter out outliers based on z-scores along columns
z_scores = np.abs(stats.zscore(growth_data, nan_policy='omit'))
growth_data = growth_data[(z_scores < 3).all(axis=1)]

growth_data = growth_data[atv_columns] / 10
ages = [int(col.split('_')[1]) for col in atv_columns]

age_columns = dict(zip(atv_columns, ages))
growth_data.rename(columns=age_columns, inplace=True)
print(growth_data)
#z = 12.3 154.9, 13.9 170.8
#input_data = [(12.3, 120.3), (13.4, 134.5), (15.6, 154.3)]
age_intervals = np.arange(8.0, 18.1, 0.1)  # Ages from 8 to 18 in 0.1 year intervals
min_age = min(growth_data.columns.astype(int))
max_age = max(growth_data.columns.astype(int))
def interpolate_row(row, age_range):
    # Create an interpolator for the row
    interpolator = interp1d(age_range, row, kind='linear', fill_value='extrapolate')

    # Generate new age intervals (0.1 year)
    new_age_range = np.arange(min_age, max_age + 0.1, 0.1)

    # Interpolate heights for these new ages
    interpolated_heights = interpolator(new_age_range)

    return new_age_range, interpolated_heights

# List to store interpolated data
interpolated_data = []

# Iterate over each row and interpolate
for index, row in growth_data.iterrows():
    ages, heights = interpolate_row(row, growth_data.columns.astype(float))
    for age, height in zip(ages, heights):
        interpolated_data.append((age, height))

# Convert to DataFrame
interpolated_growth_data = pd.DataFrame(interpolated_data, columns=['Age', 'Height'])
print(interpolated_growth_data)
#interpolated_growth_data = interpolated_growth_data.drop(columns=['child_id'])
#z = 12.3 154.9, 13.9 170.8


#def find_closest_matches(input_age_height_pairs, interpolated_growth_data, top_n=100):
    # Convert input_age_height_pairs to DataFrame
   # input_df = pd.DataFrame(input_age_height_pairs, columns=['Age', 'Height'])
    #print(input_df)
    # Prepare a DataFrame to store the distances for each input pair
    #distances_df = pd.DataFrame()
    
    # Iterate over each input pair
    #for input_age, input_height in input_age_height_pairs:
    #    interpolated_growth_data['Age'] = interpolated_growth_data['Age'].round(1)
#        input_age_rounded = round(input_age, 1)
#
#        filtered_data = interpolated_growth_data[interpolated_growth_data['Age'] == input_age_rounded]
 #       print('input age ', input_age_rounded)
        # Filter the interpolated data for the specific age
        #filtered_data = interpolated_growth_data[interpolated_growth_data['Age'] == input_age]
  #      print('filtered data',filtered_data)
    #    # If no data for this age, continue to the next pair
     #   if filtered_data.empty:
   #         continue

        # Calculate distances for the input pair to the filtered data
      #  distances = cdist(np.array([[input_age, input_height]]),
        #                  filtered_data[['Age', 'Height']].values,
       #                   'euclidean').flatten()
        
        # Store distances in the DataFrame
        #distances_df = pd.concat([distances_df, pd.DataFrame(distances, columns=[input_age], index=filtered_data.index)], axis=1)
        #print(distances_df)
    # Sum the distances across the input ages
   # total_distances = distances_df.sum(axis=1)
    #print(distances_df)
    ## Sort by total distance and get the top_n closest matches
    #closest_indices = np.argsort(total_distances)[:top_n]
    #print(closest_indices)
    #return interpolated_growth_data.loc[closest_indices]

#z = 12.3 154.9, 13.9 170.8

def find_closest_matches(input_age_height_pairs, interpolated_growth_data, top_n=100):
    # Generate the full range of ages at 0.1 year intervals
    min_age = interpolated_growth_data['Age'].min()
    max_age = interpolated_growth_data['Age'].max()
    all_ages = np.arange(min_age, max_age, 0.1)

    # Create a DataFrame for the input pattern
    input_pattern_df = pd.DataFrame(index=all_ages, columns=['Height'])
    for age, height in input_age_height_pairs:
        if age in input_pattern_df.index:
            input_pattern_df.at[age, 'Height'] = height

    # Convert the input pattern to a numpy array, filling NaNs with a large number
    input_pattern = input_pattern_df.fillna(1e10).to_numpy()
    print(input_pattern)
    # Calculate distances for each row in the interpolated data
    distances = cdist(interpolated_growth_data[['Age', 'Height']].values, input_pattern, 'euclidean')

    # Sum the distances across the input pattern
    total_distances = distances.sum(axis=1)

    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(total_distances)[:top_n]

    return interpolated_growth_data.iloc[closest_indices]


#def find_similar_growth_patterns(input_age_height_pairs, interpolated_values, top_n=100):
    # Create a mask for ages that we have input data for
    input_ages, _ = zip(*input_age_height_pairs)
    age_mask = interpolated_values.columns.isin(input_ages)
    
    # Filter out rows with missing data in the relevant age columns
    filtered_growth_data = interpolated_values.dropna(subset=interpolated_values.columns[age_mask])
    
    # Create an input pattern array, filling in with NaN where we do not have input data
    input_pattern = np.full((1, len(interpolated_values.columns)), np.nan)
    for age, height in input_age_height_pairs:
        input_pattern[0, interpolated_values.columns.get_loc(age)] = height

    # Calculate distances using only the columns for which we have input data
    distances = cdist(input_pattern[:, age_mask], filtered_growth_data.to_numpy()[:, age_mask], 'euclidean')

    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(distances[0])[:top_n]

    return filtered_growth_data.iloc[closest_indices]

def plot_growth(age_height_pairs):
    # Find the 100 most similar growth curves
    #interpolated_input = interpolated_input_data(age_height_pairs, age_intervals)
    similar_growth_curves = find_closest_matches(age_height_pairs, interpolated_growth_data)
    print(similar_growth_curves)
    # Calculate the median and the standard deviation (or interquartile range) of the heights at each age
    median_heights = similar_growth_curves.median()
    print('median heights', median_heights)
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))
    predicted_height_at_18 = median_heights.loc[18.0]

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Predicted Growth Curve Based on Similar Patterns')
    ax.set_xlabel('Age')
    ax.set_ylabel('Height (cm)')

    # Plot the median projected growth curve
    ax.plot(ages, median_heights, label='Median Projected Growth', color='blue', marker='o')

    # Plot the average cloud around the median using the standard deviation
    ax.fill_between(ages, (median_heights - iqr), (median_heights + iqr), color='skyblue', alpha=0.5, label='IQR')

    # Plot input data points
    input_ages, input_heights = zip(*age_height_pairs)
    for age, height in age_height_pairs:
        ax.scatter(input_ages, input_heights, color='red', label='Input Data Points', zorder=5)
        ax.scatter(18, predicted_height_at_18, color='green', label=f'Predicted Height at 18', zorder=5, s=100)
        ax.annotate(f'{height:.2f}', (age, height), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)


    # Annotate predicted values
    for age, height in zip(ages, median_heights):
        ax.annotate(f'{height:.2f}', (age, height), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    
    
    data_table = similar_growth_curves.to_string()
    return fig, data_table


class GrowthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Growth Prediction")

        # List to store (age, height) pairs
        self.age_height_pairs = []

        # Age input
        self.age_label = tk.Label(self, text="Age:")
        self.age_label.pack()
        self.age_entry = tk.Entry(self)
        self.age_entry.pack()

        # Height input
        self.height_label = tk.Label(self, text="Height (cm):")
        self.height_label.pack()
        self.height_entry = tk.Entry(self)
        self.height_entry.pack()

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
            self.height_entry.delete(0, tk.END)
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
        # Assuming the existence of a function that plots and returns the figure
        # such as `plot_growth` based on the previous code
        fig = plot_growth(self.age_height_pairs)
        self.display_plot(fig)

    def display_plot(self, fig):
        # If a previous figure exists, clear it before plotting a new one
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Create a new canvas and add it to the GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Display data table
        table_frame = tk.Frame(self)
        table_frame.pack(side=tk.TOP, pady=10)
        data_table_widget = tk.Text(table_frame, wrap=tk.WORD, height=5, width=30)
        data_table_widget.insert(tk.END, data_table)
        data_table_widget.config(state=tk.DISABLED)
        data_table_widget.pack()


if __name__ == "__main__":
    app = GrowthApp()
    app.mainloop()