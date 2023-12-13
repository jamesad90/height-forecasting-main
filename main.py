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

def find_similar_growth_patterns(input_age_height_pairs, interpolated_growth_data, top_n=100):
    # Generate the full range of ages at 0.1 year intervals
   # Create a mask for ages that we have input data for
    
    input_ages, _ = zip(*input_age_height_pairs)
    age_mask = interpolated_growth_data.columns.isin(input_ages)
    
    # Filter out rows with missing data in the relevant age columns
    filtered_growth_data = interpolated_growth_data.dropna(subset=interpolated_growth_data.columns[age_mask])
    
    # Create an input pattern array, filling in with NaN where we do not have input data
    input_pattern = np.full((1, len(interpolated_growth_data.columns)), np.nan)
    #print('input pattern', input_pattern)
    for age, height in input_age_height_pairs:
        input_pattern[0, interpolated_growth_data.columns.get_loc(age)] = height
    #print('input pattern', input_pattern)
    # Calculate distances using only the columns for which we have input data
    distances = cdist(input_pattern[:, age_mask], filtered_growth_data.to_numpy()[:, age_mask], 'euclidean')
    #print('distances:',distances)
    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(distances[0])[:top_n]

    return filtered_growth_data.iloc[closest_indices]

interpolated_ages = np.arange(8.0, 19.0, 0.1) 
def plot_growth(age_height_pairs, interpolated_growth_data):
    # Find the 100 most similar growth curves
   
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, interpolated_growth_data)

    # Calculate the median and the standard deviation (or interquartile range) of the heights at each age
    median_heights = similar_growth_curves.median()
    #print(median_heights)
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))
    predicted_height_at_18 = median_heights.loc[18.0].round(1)
    print(predicted_height_at_18)
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
        ax.scatter(18.0, predicted_height_at_18, color='green', label=f'Predicted Height at 18', zorder=5, s=100)
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


def process_reference_data(csv_file_path):
    data = csv_file_path
    atv_columns = [col for col in data.columns if col.startswith('ATV_')]
    id_column = 'child_id' if 'child_id' in data.columns else None

    long_format_data = pd.melt(data, id_vars=id_column, value_vars=atv_columns, var_name='Age', value_name='Height')
    long_format_data['Age'] = long_format_data['Age'].apply(lambda x: float(x.split('_')[1]))
    long_format_data = long_format_data.dropna(subset=['Height'])
    long_format_data.dropna(subset=['Age', 'Height'], inplace=True)

    growth_data = long_format_data
    growth_data['Height'] = growth_data['Height'] / 10
    growth_data['Height'] = growth_data['Height'].round(1)
    growth_data['Age'] = growth_data['Age'].round(1)
    long_format_data = growth_data

    interpolated_ages = np.arange(8.0, 19.0, 0.1)

    interpolated_list = []
    for child_id in long_format_data['child_id'].unique():
        child_data = long_format_data[long_format_data['child_id'] == child_id]
        interpolator = interp1d(child_data['Age'], child_data['Height'], kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_heights = interpolator(interpolated_ages)
        df_temp = pd.DataFrame({'child_id': child_id, 'Age': interpolated_ages, 'Height': interpolated_heights})
        interpolated_list.append(df_temp)

    interpolated_data = pd.concat(interpolated_list, ignore_index=True)
    interpolated_data['Age'] = interpolated_data['Age'].round(1)

    wide_format_data = interpolated_data.pivot(index='child_id', columns='Age', values='Height')
    wide_format_data.reset_index(inplace=True)
    growth_data = wide_format_data
    #growth_data = wide_format_data.drop(columns=['child_id'])
    #print(growth_data)
    return growth_data

def predict_heights(age_height_pairs, growth_data):
    # Find the 100 most similar growth curves
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, growth_data)
    median_heights = similar_growth_curves.median()
    if not isinstance(median_heights, pd.DataFrame):
        median_heights = median_heights.to_frame()
    median_heights = median_heights.squeeze()
    #print(median_heights)
    # Calculate the median and the IQR of the heights at each age
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))

    # Predict height at age 18
    predicted_height_at_18 = median_heights.loc[18.0] if 18.0 in median_heights.index else None

    return median_heights, iqr, predicted_height_at_18
