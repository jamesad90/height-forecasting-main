import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

# Load your data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Drop 'child_id' column
data = data.drop(columns=['child_id'])

# Extract ATV_ columns
atv_columns = [col for col in data.columns if 'ATV_' in col]

# Divide these columns by 10 if necessary
data[atv_columns] = data[atv_columns] / 10

# Reshape data to long format with 'Age' and 'Height' columns
melted_data = pd.melt(data, id_vars=[], value_vars=atv_columns, var_name='Age', value_name='Height')

# Convert 'Age' from 'ATV_xx' format to integer
melted_data['Age'] = melted_data['Age'].apply(lambda x: int(x.split('_')[1]))

# Filter out outliers based on z-scores
z_scores = np.abs(stats.zscore(melted_data['Height'], nan_policy='omit'))
melted_data = melted_data[(z_scores < 3)]

# Rename melted_data to growth_data for clarity
growth_data = melted_data

input_data = [(12.3, 120.3), (13.4, 134.5), (15.6, 154.3)]
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


def find_closest_matches(interpolated_input_data, growth_data, top_n=100):
    closest_matches = []

    for age, height in interpolated_input_data:
        # Calculate distances for each input point
        distances = cdist(np.array([(age, height)]), growth_data[['Age', 'Height']].values, 'euclidean')

        # Sort by distance and get the top_n closest matches
        closest_indices = np.argsort(distances[0])[:top_n]
        closest_points = growth_data.iloc[closest_indices]
        closest_matches.append(closest_points)
        print(closest_matches)
    return closest_matches


unique_ages = growth_data['Age'].unique()
interpolated_input = interpolate_input_data(input_data, age_intervals)

closest_matches = find_closest_matches(interpolated_input, growth_data)
