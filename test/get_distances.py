def find_similar_growth_patterns(input_age_height_pairs, growth_data, top_n=100):
    # Create a mask for ages that we have input data for
    input_ages, _ = zip(*input_age_height_pairs)
    age_mask = growth_data.columns.isin(input_ages)
    
    # Filter out rows with missing data in the relevant age columns
    filtered_growth_data = growth_data.dropna(subset=growth_data.columns[age_mask])
    
    # Create an input pattern array, filling in with NaN where we do not have input data
    input_pattern = np.full((1, len(growth_data.columns)), np.nan)
    for age, height in input_age_height_pairs:
        input_pattern[0, growth_data.columns.get_loc(age)] = height

    # Calculate distances using only the columns for which we have input data
    distances = cdist(input_pattern[:, age_mask], filtered_growth_data.to_numpy()[:, age_mask], 'euclidean')

    # Get indices of the top_n closest growth curves
    closest_indices = np.argsort(distances[0])[:top_n]

    return filtered_growth_data.iloc[closest_indices]


def plot_growth(age_height_pairs):
    # Find the 100 most similar growth curves
    similar_growth_curves = find_similar_growth_patterns(age_height_pairs, growth_data)

    # Calculate the median and the standard deviation (or interquartile range) of the heights at each age
    median_heights = similar_growth_curves.median()
    iqr = np.subtract(*np.percentile(similar_growth_curves, [75, 25], axis=0))
    predicted_height_at_18 = median_heights.loc[18]

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
