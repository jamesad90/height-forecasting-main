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
import matplotlib.lines as mlines
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

interpolated_ages = np.arange(8.0, 18.10, 0.1)[:-1] 
#print(interpolated_ages)

def generate_predicted_heights(age_height_pairs, similar_growth_curves):
    """
    Generates predicted heights based on input data and average yearly growth.
    
    :param age_height_pairs: List of tuples with age and height.
    :param similar_growth_curves: DataFrame with height data.
    :return: Series with predicted heights from age 8.0 to 18.0.
    """
    # Calculate average yearly growth
   
    yearly_growth = similar_growth_curves.diff(axis=1).iloc[:, 0::10] *10
    
    average_yearly_growth = yearly_growth.median(axis=0)
    new_avg_index = average_yearly_growth.index - 1
    new_index = yearly_growth.index -1
    #average_yearly_growth = average_yearly_growth.copy()
    average_yearly_growth.index = new_avg_index
    yearly_growth.index = new_index
    #print(yearly_growth)
    #print(average_yearly_growth)
    # Initialize a Series to store predicted heights
    age_range = np.arange(8.0, 18.1, 1.0)
    predicted_heights = pd.Series(index=age_range, name='Height')

    # Fill in the input data
    for age, height in age_height_pairs:
        age = round(age)
        predicted_heights.at[age] = height

    # Predict future and past heights
    for age in predicted_heights.index:
        if pd.isna(predicted_heights.at[age]):  # Only predict if height is not already provided
            # Use growth rate from previous year (offset by -1)
            if age < min(age_height_pairs)[0]:  # Predict past heights
                growth_rate = average_yearly_growth.get(age, 0)
                print(growth_rate)
                predicted_heights.at[age] = predicted_heights.at[age + 1.0] - growth_rate
            elif age > max(age_height_pairs)[0]:  # Predict future heights
                growth_rate = average_yearly_growth.get(age - 1.0, 0)
                predicted_heights.at[age] = predicted_heights.at[age - 1.0] + growth_rate
    #print(predicted_heights)
    return predicted_heights, yearly_growth


def plot_growth(age_height_pairs, interpolated_growth_data):
    # Find the 100 most similar growth curves
   
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, interpolated_growth_data)
    average_yearly_growth, yearly_growth  = generate_predicted_heights(age_height_pairs, similar_growth_curves)
   
    # Calculate the median and the standard deviation (or interquartile range) of the heights at each age
    median_heights = similar_growth_curves.median()
    #average_yearly_growth = similar_growth_curves
    #print(median_heights)
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))
    avg_iqr = np.subtract(*np.percentile(yearly_growth, [75, 25], axis=0))
    #print(avg_iqr)
    #print(iqr)
    predicted_height_at_18 = median_heights.loc[18.0].round(1)
   # print(predicted_height_at_18)
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Predicted Growth Curve Based on Similar Patterns')
    ax.set_xlabel('Age')
    ax.set_ylabel('Height (cm)')
    plt.xticks(np.arange(8, 98, 1))
    plt.yticks(np.arange(100, 250, 5))
    #print(median_heights)
    # Plot the median projected growth curve
    ax.plot(interpolated_ages, median_heights, label='Median Projected Growth', color='blue', marker='o')
    
    # Plot the average cloud around the median using the standard deviation
    ax.fill_between(interpolated_ages, (median_heights - iqr), (median_heights + iqr), color='skyblue', alpha=0.5, label='Similar Curves IQR')

    # Plot input data points
    input_ages, input_heights = zip(*age_height_pairs)
    #print('next 2:',input_ages, input_heights)
    for age, height in age_height_pairs:
        ax.scatter(input_ages, input_heights, color='black', label='Input Data Points', zorder=5, marker = 's')
        #ax.scatter(18.0, predicted_height_at_18, color='green', label=f'Predicted Height at 18', zorder=5, s=100)
        ax.annotate(f'{height:.2f}',(age, height), 
                    textcoords="offset points", 
                    xytext=(0, 70), ha='center', 
                    fontsize=12, 
                    fontweight='bold', 
                    backgroundcolor = 'yellow')
        ax.scatter
    annotation_proxy = mlines.Line2D([], [], color='yellow', marker='o', markersize=10, label='Input Data')

    # Annotate predicted values
    whole_number_ages = np.arange(8.0, 19.0, 1.0)
    for age in whole_number_ages:
        if age in median_heights.index:
            height = median_heights.loc[age]
            #print(age)
            #print(height)
            ax.annotate(f'{height:.2f}', (age, height), 
                        textcoords="offset points", 
                        xytext=(0, 20), 
                        ha='center', 
                        fontsize=12, 
                        color = 'blue')
    for age in whole_number_ages:
        if age in average_yearly_growth.index:
            height = average_yearly_growth.loc[age]
            #print(age)
            #print(height)
            ax.annotate(f'{height:.2f}', (age, height), 
                        textcoords="offset points", 
                        xytext=(0, -20), 
                        ha='center', 
                        fontsize=12, 
                        color = 'red')
    #print(average_yearly_growth)
    ax.plot(whole_number_ages, average_yearly_growth, label='Average Yearly Growth', color='red', marker='o')
    ax.fill_between(whole_number_ages, (average_yearly_growth - avg_iqr), (average_yearly_growth + avg_iqr), color='red', alpha=0.5, label='Avg Growth IQR')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(annotation_proxy) 
    #by_label = dict(zip(labels, handles))
    ax.legend(handles=handles, labels=labels)
    
    
    #data_table = similar_growth_curves.to_string()
    return fig #data_table

def process_reference_data_test(csv_file_path):
    data = pd.read_csv(csv_file_path)
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

    interpolated_ages = np.arange(8.0, 18.10, 0.1)[:-1] 

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
    growth_data = wide_format_data.drop(columns=['child_id'])
    #print(growth_data)
    return growth_data
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

    interpolated_ages = np.arange(8.0, 18.10, 0.1)[:-1] 

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

def predict_heights(age_height_pairs, interpolated_growth_data):
    # Find the 100 most similar growth curves
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, interpolated_growth_data)
    #print(similar_growth_curves)
    
    average_yearly_growth  = generate_predicted_heights(age_height_pairs, similar_growth_curves)
    median_heights = similar_growth_curves.median()
    if not isinstance(median_heights, pd.DataFrame):
        median_heights = median_heights.to_frame()
    median_heights = median_heights.squeeze()
    #print(median_heights)
    if not isinstance(average_yearly_growth, pd.DataFrame):
        average_yearly_growth = average_yearly_growth.to_frame()
    average_yearly_growth.squeeze()
    # Calculate the median and the IQR of the heights at each age
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))

    # Predict height at age 18
    predicted_height_at_18 = median_heights.loc[18.0] if 18.0 in median_heights.index else None
    predicted_height_at_18_ii = average_yearly_growth.loc[18.0] if 18.0 in average_yearly_growth.index else None
    #print(predicted_height_at_18, predicted_height_at_18_ii)
    return median_heights, iqr, predicted_height_at_18, average_yearly_growth, predicted_height_at_18_ii


