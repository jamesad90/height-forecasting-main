import pandas as pd
# Load the existing combination_results.csv into a DataFrame
results_df = pd.read_csv('combination_results.csv')



# Group by the new 'Group' column and calculate the mean for 'Predicted Height at 18' and 'MAE'
average_values_df = results_df.groupby('Ages Used')['Actual Height at 18','Predicted Height at 18', 'MAE'].mean().reset_index()

# Save the average_values_df DataFrame to a CSV file
average_values_csv_path = 'average_values.csv'
average_values_df.to_csv(average_values_csv_path, index=False)