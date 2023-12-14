from scipy import stats
import pandas as pd
import numpy as np
from main import find_similar_growth_patterns, process_reference_data, predict_heights, plot_growth
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from itertools import combinations
csv_file_path = 'svk_height_weight_mens_2008_v2.csv'
reference_data = process_reference_data(csv_file_path)

# Load the dataset and display the first few rows to understand its structure
test_data = pd.read_csv('Berkeley_EB_4.csv')
#print(test_data.head())


wide_format_test_data = test_data.pivot(index='child_id', columns='age_decimalyears', values='height_cm')

# Display the first few rows of the wide format data
#print(wide_format_test_data.head())
#wide_format_test_data = wide_format_test_data.reset_index()


def calculate_accuracy(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    return mae
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))

# Assuming reference_data is your reference dataset for the model
# Replace 'reference_data' with your actual reference dataset

rmse_results = {}
mae_results = {}
comparison_data = []

input_data = [(9, 139.4), (10, 144.9), (11, 149.9)]


plot_growth(input_data,reference_data)

#predict_heights(input_data, reference_data)
#similar_growth_curves = find_similar_growth_patterns(input_data, reference_data)
#median_heights = similar_growth_curves.median()
#print(median_heights)

def process_combinations(child_id, combo, row, wide_format_test_data):
    # Convert combo to age_height_pairs
    age_height_pairs = [(age, row[age]) for age in combo if pd.notna(row[age])]
    if age_height_pairs:
        # Predict the height at 18 using the model
        _, _, predicted_height_at_18 = predict_heights(age_height_pairs, reference_data)
        actual_height_at_18 = row.get(18.0, np.nan)
        # Calculate MAE only if actual height at 18 is not NaN
        mae = np.nan
        if not np.isnan(actual_height_at_18):
            mae = calculate_accuracy([actual_height_at_18], [predicted_height_at_18])
        return {
            'child_id': child_id,
            'Ages Used': combo,
            'Num Observations': len(combo),
            'Predicted Height at 18': predicted_height_at_18,
            'Actual Height at 18': actual_height_at_18,
            'MAE': mae
        }

# Set the age range from which to create combinations
age_range = np.arange(9, 18, 1.0)

# Create a DataFrame to hold the results
results = []

for child_id, row in tqdm(wide_format_test_data.iterrows()):
    #child_id = row['child_id']
    # Generate all non-empty combinations of the ages in the age range for this individual
    for r in tqdm(range(1, len(age_range) + 1)):
        for combo in tqdm(combinations(age_range, r)):
            result = process_combinations(child_id, combo, row, wide_format_test_data)
            if result:
                results.append(result)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv('combination_results.csv', index=False)