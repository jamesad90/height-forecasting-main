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
root = tk.Tk()
root.withdraw()  # Hide the main window

csv_file_path = filedialog.askopenfilename(title="Select the CSV file",
                                           filetypes=[("CSV files", "*.csv")])
if not csv_file_path:
    raise Exception("A CSV file must be selected to proceed.")

# Load your data
data = pd.read_csv(csv_file_path)

# Assume that columns starting with 'ATV_' are age-related columns
# and 'child_id' is an identifier column (if present)
atv_columns = [col for col in data.columns if col.startswith('ATV_')]
id_column = 'child_id' if 'child_id' in data.columns else None

# Reshape the data to long format
long_format_data = pd.melt(data, id_vars='child_id', value_vars=atv_columns, var_name='Age', value_name='Height')

# Optionally, convert 'Age' from 'ATV_xx' format to numeric
long_format_data['Age'] = long_format_data['Age'].apply(lambda x: float(x.split('_')[1]))

# Drop NaN values if necessary
long_format_data = long_format_data.dropna(subset=['Height'])

# Check if there are any non-numeric values or NaNs
#print(long_format_data['Age'].isnull().sum())  # Number of NaNs in the 'Age' column

# Drop rows where 'Age' or 'Height' is NaN, if needed
long_format_data.dropna(subset=['Age', 'Height'], inplace=True)

# Optionally, convert 'Age' to int if it only contains integer values
#long_format_data['Age'] = long_format_data['Age'].astype(float)


growth_data = long_format_data
growth_data['Height']= growth_data['Height']/10
growth_data['Height']=growth_data['Height'].round(1)
growth_data['Age']= growth_data['Age'].round(1)
#print(growth_data)
long_format_data = growth_data

interpolated_ages = np.arange(8.0, 19.0, 0.1)  # Ages from 8 to 18 in 0.1 year intervals

# Create a new DataFrame to store interpolated values
interpolated_data = pd.DataFrame()

# Get the range of ages
min_age = long_format_data['Age'].min()
max_age = long_format_data['Age'].max()
#interpolated_ages = np.arange(min_age, max_age, 0.1)
print('next')
# Interpolate for each child
# Initialize an empty list to store the interpolated data for each child
interpolated_list = []

# Iterate over each child
for child_id in long_format_data['child_id'].unique():
    child_data = long_format_data[long_format_data['child_id'] == child_id]

    # Create an interpolator using the child's data
    # Assuming child_data has 'Age' and 'Height' columns
    interpolator = interp1d(child_data['Age'], child_data['Height'],
                            kind='linear', bounds_error=False, fill_value='extrapolate')

    # Interpolate the heights for the specified age range
    interpolated_heights = interpolator(interpolated_ages)

    # Create a DataFrame for the current child and append it to the list
    df_temp = pd.DataFrame({
        'child_id': child_id,
        'Age': interpolated_ages,
        'Height': interpolated_heights
    })
    interpolated_list.append(df_temp)

# Concatenate all the dataframes in the list into a single DataFrame
interpolated_data = pd.concat(interpolated_list, ignore_index=True)
interpolated_data['Age'] = interpolated_data['Age'].round(1)
print('next')
# Reset the index
#interpolated_data.reset_index(drop=True, inplace=True)

print(interpolated_data)
wide_format_data = interpolated_data.pivot(index='child_id', columns='Age', values='Height')

# Resetting the index to make 'child_id' a column again
wide_format_data.reset_index(inplace=True)
print(wide_format_data)
growth_data = wide_format_data.drop(columns=['child_id'])

def find_similar_growth_patterns(input_age_height_pairs, interpolated_growth_data, top_n=100):
    # Generate the full range of ages at 0.1 year intervals
   # Create a mask for ages that we have input data for
    input_ages, _ = zip(*input_age_height_pairs)
    age_mask = growth_data.columns.isin(input_ages)
    
    # Filter out rows with missing data in the relevant age columns
    filtered_growth_data = growth_data.dropna(subset=growth_data.columns[age_mask])
    
    # Create an input pattern array, filling in with NaN where we do not have input data
    input_pattern = np.full((1, len(growth_data.columns)), np.nan)
    print('input pattern', input_pattern)
    for age, height in input_age_height_pairs:
        input_pattern[0, growth_data.columns.get_loc(age)] = height
    print('input pattern', input_pattern)
    # Calculate distances using only the columns for which we have input data
    distances = cdist(input_pattern[:, age_mask], filtered_growth_data.to_numpy()[:, age_mask], 'euclidean')
    print('distances:',distances)
    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(distances[0])[:top_n]

    return filtered_growth_data.iloc[closest_indices]


def plot_growth(age_height_pairs):
    # Find the 100 most similar growth curves
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, growth_data)

    # Calculate the median and the standard deviation (or interquartile range) of the heights at each age
    median_heights = similar_growth_curves.median()
    print(median_heights)
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))
    predicted_height_at_18 = median_heights.loc[18.0]

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Predicted Growth Curve Based on Similar Patterns')
    ax.set_xlabel('Age')
    ax.set_ylabel('Height (cm)')

    # Plot the median projected growth curve
    ax.plot(interpolated_ages, median_heights, label='Median Projected Growth', color='blue', marker='o')

    # Plot the average cloud around the median using the standard deviation
    ax.fill_between(interpolated_ages, (median_heights - iqr), (median_heights + iqr), color='skyblue', alpha=0.5, label='IQR')

    # Plot input data points
    input_ages, input_heights = zip(*age_height_pairs)
    for age, height in age_height_pairs:
        ax.scatter(input_ages, input_heights, color='red', label='Input Data Points', zorder=5)
        ax.scatter(18, predicted_height_at_18, color='green', label=f'Predicted Height at 18', zorder=5, s=100)
        ax.annotate(f'{height:.2f}', (age, height), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)


    # Annotate predicted values
    whole_number_ages = np.arange(8.0, 19.0, 1.0)
    for age in whole_number_ages:
        if age in median_heights.index:
            height = median_heights.loc[age]
            print(age)
            print(height)
            ax.annotate(f'{height:.2f}', (age, height), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)

    
    
    data_table = similar_growth_curves.to_string()
    return fig, data_table

#11.8 129.8, 12.7 145.4, 15.4 162.1
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
        fig, data_table = plot_growth(self.age_height_pairs)
        self.display_plot(fig, data_table)

    def display_plot(self, fig, data_table):
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