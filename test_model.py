from scipy import stats
import pandas as pd
import numpy as np
from main import find_similar_growth_patterns, process_reference_data, predict_heights
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from itertools import combinations
csv_file_path = 'svk_height_weight_mens_2008_v2.csv'
reference_data = process_reference_data(csv_file_path)
print('next')
# Load the dataset and display the first few rows to understand its structure
test_data = pd.read_csv('Berkeley_EB_4.csv')
print(test_data.head())


wide_format_test_data = test_data.pivot(index='child_id', columns='age_decimalyears', values='height_cm')

# Display the first few rows of the wide format data
print(wide_format_test_data.head())
#wide_format_test_data = wide_format_test_data.reset_index()

print('next')
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
z=z






z=z
for start_age in tqdm(np.arange(9, 18, 1.08)):
    for child_id, row in tqdm(wide_format_test_data.iterrows()):
        # Get actual height at age 18 for this child
        actual_height_at_18 = row.get(18.0, np.nan)  # Use .get to avoid KeyError if 18.0 is not present
        print(child_id)
        for end_age in tqdm(np.arange(start_age, 18.1, 0.1)):
            #print(start_age,end_age)
            # Subset the row for the current age range and drop NaN values
            observations = row.loc[start_age:end_age].dropna()
            num_observations = len(observations)

            # Prepare the input for the model
            input_age_height_pairs = list(zip(observations.index, observations.values))

            # Use the model to predict heights
            _, _, predicted_height_at_18 = predict_heights(input_age_height_pairs, reference_data)
            print(predicted_height_at_18)
            # Calculate MAE only if actual height at 18 is not NaN
            if not np.isnan(actual_height_at_18):
                mae = calculate_accuracy([actual_height_at_18], [predicted_height_at_18])
            else:
                mae = np.nan  # Or some placeholder if actual height is not known

            # Append the results to the comparison data list
            comparison_data.append({
                'Start Age': start_age,
                'End Age': end_age,
                'Num_observations': num_observations,
                'Predicted_height_at_18': predicted_height_at_18,
                'Actual_height_at_18': actual_height_at_18,
                'MAE': mae
            })

# Create a DataFrame from the comparison data
comparison_df = pd.DataFrame(comparison_data)

# Save the DataFrame to a CSV file
comparison_df.to_csv('comparison_data.csv', index=False)


z=z


for age in tqdm(np.arange(9, 18.1, 0.1)):  # Assuming ages are from 9 to 18
    all_rmse = []
    all_mae = []
    for index, row in tqdm(wide_format_test_data.iterrows()):
        numeric_index = pd.to_numeric(row.index, errors='coerce')

        # Prepare age-height pairs up to the current age
        age_height_pairs = [(a, row[a]) for a in numeric_index if a <= age and pd.notna(row[a])]
        print(age_height_pairs)
        # Predict heights using the model
        predicted_heights = predict_heights(age_height_pairs, reference_data)
        #print(predicted_heights)
        # Ensure predicted_heights is a Series
        if isinstance(predicted_heights, pd.Series):
            # Actual heights for the same age range
            actual_heights = row[row.index.isin([a for a, _ in age_height_pairs])]

            # Store the actual and predicted values for analysis
            for a in actual_heights.index:
                comparison_data.append({
                    'id': index,
                    'age': a,
                    'actual_height': actual_heights[a],
                    'predicted_height': predicted_heights[a],
                    'age_limit': age
                })

            # Calculate RMSE and store it
            rmse = calculate_rmse(actual_heights, predicted_heights[actual_heights.index])
            all_rmse.append(rmse)
            #Calculate and store MAE
            mae = calculate_rmse(actual_heights, predicted_heights[actual_heights.index])
            all_mae.append(mae) 

    # Average RMSE for this age
    rmse_results[age] = np.mean(all_rmse)
    mae_results[age] = np.mean(all_mae)
# Creating a DataFrame from the stored data
comparison_df = pd.DataFrame(comparison_data)

# Saving RMSE results and comparison data to CSV files
#comparison_df.to_csv('height_comparison.csv', index=False)
#pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE']).to_csv('rmse_results.csv')
pd.DataFrame.from_dict(mae_results, orient='index', columns=['MAE']).to_csv('mae_results.csv')
z=z





# Initialize a dictionary to store accuracy results
accuracy_results = {}


# Iterate through a range of years (number of observations)
for num_years in range(1, len(wide_format_test_data.columns) + 1):
    all_mae = []

    for index, row in wide_format_test_data.iterrows():
        # Prepare input_age_height_pairs with increasing number of years
        input_age_height_pairs = [(age, row[age]) for age in row.index[:num_years] if pd.notna(row[age])]

        # Apply the model
        # Here, replace 'find_similar_growth_patterns' and 'reference_data' with your actual model and data
        #similar_growth_curves = find_similar_growth_patterns(input_age_height_pairs, reference_data)

        # Calculate the predicted heights - This will depend on how your model provides predictions
        # For demonstration, I'm assuming a function 'get_predicted_heights' that gives predicted heights
        # Replace it with the actual way of obtaining predictions from your model
        median_heights, iqr, predicted_height_at_18 = predict_heights(input_age_height_pairs, reference_data)


        # Calculate accuracy - here using MAE as an example
        true_heights = row.dropna().values
        mae = calculate_accuracy(true_heights, median_heights)
        all_mae.append(mae)

    # Average MAE for this number of years
    average_mae = np.mean(all_mae)
    accuracy_results[num_years] = average_mae

# Print or analyze the accuracy_results
print(accuracy_results)