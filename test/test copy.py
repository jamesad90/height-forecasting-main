import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

# Load your data
# Drop 'child_id' column
# Load the CSV file
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

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
print(long_format_data['Age'].isnull().sum())  # Number of NaNs in the 'Age' column

# Drop rows where 'Age' or 'Height' is NaN, if needed
long_format_data.dropna(subset=['Age', 'Height'], inplace=True)

# Optionally, convert 'Age' to int if it only contains integer values
#long_format_data['Age'] = long_format_data['Age'].astype(float)


growth_data = long_format_data
growth_data['Height']= growth_data['Height']/10
growth_data['Height']=growth_data['Height'].round(1)
growth_data['Age']= growth_data['Age'].round(1)
print(growth_data)
age_intervals = np.arange(8, 18.1, 0.1)  # Ages from 8 to 18 in 0.1 year intervals

def interpolate_input_data(input_data, age_intervals):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, columns=['Age', 'Height'])

    # Ensure ages are sorted
    input_df.sort_values(by='Age', inplace=True)

    # Create an interpolator
    interpolator = interp1d(input_df['Age'], input_df['Height'], kind='linear', bounds_error=False, fill_value="extrapolate")

    # Interpolate for the defined age intervals
    interpolated_heights = interpolator(age_intervals)

    return list(zip(age_intervals, interpolated_heights))

 # Ages from 8 to 18 in 0.1 year intervals
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
interpolated_growth_data = pd.DataFrame(interpolated_data, columns=['Age', 'Height']).reset_index()
#print(interpolated_growth_data)


def find_closest_matches(input_age_height_pairs, interpolated_growth_data, top_n=100):
    # Generate the full range of ages at 0.1 year intervals
    min_age = interpolated_growth_data['Age'].min()
    max_age = interpolated_growth_data['Age'].max()
    all_ages = np.arange(min_age, max_age, 0.1)

    # Create a DataFrame for the input pattern
    input_pattern_df = pd.DataFrame(index = all_ages, columns=['Height']).round(1)
    input_pattern_df['Age'] = all_ages.round(1)
    print(input_pattern_df)
    for age, height in input_age_height_pairs:
        print('ages and heights', age, height)
        if age in input_pattern_df.index:
            input_pattern_df.at[age, 'Height'] = height
    print(input_pattern_df)
    # Convert the input pattern to a numpy array, filling NaNs with a large number
    input_pattern = input_pattern_df.fillna(np.nan).to_numpy().round(1)
    print(input_pattern)
    # Calculate distances for each row in the interpolated data
    distances = cdist(interpolated_growth_data[['Age', 'Height']].values, input_pattern, 'euclidean')

    # Sum the distances across the input pattern
    #total_distances = distances.sum(axis=1)

    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(distances)[:top_n]

    return interpolated_growth_data.iloc[closest_indices]

input_data = [(8.1, 100.3),(12.3, 120.3), (13.4, 134.5), (15.6, 154.3)]
#closest_matches = find_closest_matches(input_data, interpolated_growth_data) 


def plot_growth(age_height_pairs):
    # Find the 100 most similar growth curves
    similar_growth_curves = find_closest_matches(input_data, interpolated_growth_data)

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

plot_growth(input_data)