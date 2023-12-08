import os
os.chdir('src')
import os
from scipy import stats
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load your data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Convert height measurements from mm to cm
height_columns = [col for col in data.columns if col.startswith('ATV_')]
data[height_columns] = data[height_columns] / 10

# Melt the DataFrame
melted_data = pd.melt(data, id_vars=['child_id'], value_vars=height_columns, var_name='Age', value_name='Height')

# Convert age to integer
melted_data['Age'] = melted_data['Age'].str.extract('(\d+)').astype(int)

# Filter out outliers using z-scores
z_scores = stats.zscore(melted_data['Height'])
melted_data = melted_data[(np.abs(z_scores) < 3)]

# Group data by child_id
grouped = melted_data.groupby('child_id')

# Define a function to fit a growth curve to the data
def growth_curve(age, age_at_peak_velocity, peak_velocity, pre_pubertal_gain_rate, post_pubertal_gain_rate, age_at_adult_height):
    # Calculate the pre-pubertal growth using a logistic function
    pre_pubertal_growth = (max_height * pre_pubertal_gain_rate) / (1 + np.exp(-pre_pubertal_gain_rate * (age - age_at_peak_velocity)))

    # Calculate the peak velocity phase using a logistic function
    peak_velocity_growth = peak_velocity * (1 / (1 + np.exp(-0.5 * (age - age_at_peak_velocity))))

    # Calculate the post-pubertal growth as a reverse logistic curve approaching max height
    post_pubertal_growth = (max_height * post_pubertal_gain_rate) * (1 - 1 / (1 + np.exp(-post_pubertal_gain_rate * (age - age_at_adult_height))))

    # Sum the growth components, ensuring it doesn't exceed max height
    total_height = pre_pubertal_growth + peak_velocity_growth + post_pubertal_growth
    total_height = np.minimum(total_height, max_height)

    return total_height


# Dictionary to hold the extracted features for each child
features_dict = {}
from sklearn.linear_model import LinearRegression
# Loop through each child's data
for child_id, group_data in grouped:
    age_data = group_data['Age'].values
    height_data = group_data['Height'].values
    pre_pubertal_data = group_data[(group_data['Age'] <= 8) & (group_data['Age'] >= 2)]  # Example age range for pre-pubertal

    # Prepare the data for modeling
    t = pre_pubertal_data['Age'].values.reshape(-1, 1)  # Age data
    s = pre_pubertal_data['Height'].values  # Height data

    # Fit the linear regression model
    regression_model = LinearRegression()
    regression_model.fit(t, s)

    # The slope of the regression line represents the pre_pubertal_gain_rate
    pre_pubertal_gain_rate = regression_model.coef_[0]
    # Assuming 'data' is a DataFrame with 'age' and 'height' columns for post-pubertal ages
    post_pubertal_data = group_data[(group_data['Age'] >= 16) & (group_data['Age'] <= 20)]  # Example age range for post-pubertal

    # Prepare the data for modeling
    X = post_pubertal_data['Age'].values.reshape(-1, 1)  # Age data
    y = post_pubertal_data['Height'].values  # Height data

    # Fit the linear regression model
    regression_model = LinearRegression()
    regression_model.fit(X, y)

    # The slope of the regression line represents the post_pubertal_gain_rate
    post_pubertal_gain_rate = regression_model.coef_[0]

    # Skip the iteration if there's not enough data
    if len(age_data) < 3:
        continue

    

    # Fit a spline to estimate the first and second derivatives
    spline = UnivariateSpline(age_data, height_data, k=4, s=0)
    first_derivative = spline.derivative(n=1)
    second_derivative = spline.derivative(n=2)

    # Estimate peak height velocity (PHV)
    velocities = first_derivative(age_data)
    phv_age = age_data[np.argmax(velocities)]
    phv_value = np.max(velocities)
    final_height = np.max(height_data)

    # Calculate acceleration at PHV age
    acceleration = second_derivative(phv_age)

    # Store the extracted features in the dictionary
    features_dict[child_id] = {
        'PHV_Age': phv_age,
        'PHV_Value': phv_value,
        'pre_pubertal_gain_rate': pre_pubertal_gain_rate,
        'Post_pubertal_gain_rate': post_pubertal_gain_rate,
        'Age at adult height': final_height,
        'child_id': child_id
    }

n_clusters = 6
# Now 'features_dict' contains the extracted features for each child
# Convert the features_dict into a DataFrame
features_df = pd.DataFrame.from_dict(features_dict, orient='index')
#print(features_df)
# Melt the features_df to match the structure of the melted_data
#parameter_names = ['PHV_Age', 'PHV_Value', 'Acceleration', 'Curve_Param_a', 'Curve_Param_b', 'Curve_Param_c']


# Fit the K-Means clustering model to the parameters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans = kmeans.fit(features_df)
features_df['ClusterLabel'] = kmeans.labels_

from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.5f' % x)
# Train your KMeans model (kmeans) on melted_features_df

# Function to fit a growth curve to the data
#def growth_curve(age, a, b, c):
#    # Here you can define the form of the growth curve, such as a logistic curve
#    return a / (1 + np.exp(b - c * age))



def predict_and_plot_growth_data(user_data, melted_data, kmeans, growth_curve):
    predicted_growth_data = pd.DataFrame(columns=['Age', 'Height'])
    closest_matching_data = pd.DataFrame()  # DataFrame to store closest matching data points
    
    for user_data_point in user_data:
        user_age, user_height = user_data_point  # Assuming user_data includes participant IDs

        # Calculate distances from the user-provided data point to all points in the melted data
        distances = cdist(np.array([[user_age, user_height]]), melted_data[['Age', 'Height']], metric='euclidean')[0]
        closest_data_index = np.argmin(distances)

        # Extract the closest matching data point and store it
        closest_data_point = melted_data.iloc[[closest_data_index]]
        closest_matching_data = pd.concat([closest_matching_data, closest_data_point], ignore_index=True)
        # Find the participant ID for the closest data point
        participant_id = melted_data.iloc[closest_data_index]['child_id']

        # Get the corresponding cluster label for the participant from the unmelted data
        participant_row = features_df[features_df['child_id'] == participant_id]
        if not participant_row.empty:
            closest_cluster_label = participant_row['ClusterLabel'].iloc[0]

             # Get the cluster label for the closest data point
            
            closest_cluster_features = kmeans.cluster_centers_[closest_cluster_label]
            print(closest_cluster_features)
            # Predict heights using the growth curve and cluster center features
            ages = np.arange(user_age, 19)
            a = 153.7
            b = -13654.54
            c = 2527
            heights = growth_curve(ages, closest_cluster_features[:4])
            heights = growth_curve(ages, a, b, c)
            # Append the predicted data to the DataFrame
            age_height_df = pd.DataFrame({'Age': ages, 'Height': heights})
            predicted_growth_data = pd.concat([predicted_growth_data, age_height_df], ignore_index=True)
        else:
            # Handle the case where participant_id is not found in unmelted_data
            continue
        # Plot the user data, predicted data, and closest matching data
    # Plot the user data, predicted data, and closest matching data
    plt.figure(figsize=(10, 6))

    # Plot closest matching data points, using child_id for coloring or labeling
    plt.scatter(closest_matching_data['Age'], closest_matching_data['Height'],
                c=closest_matching_data['child_id'], label='Closest Matching Data', cmap='viridis')

    # Plot each user data point
    for user_data_point in user_data:
        user_age, user_height = user_data_point
        plt.scatter(user_age, user_height, label=f'User Data (Participant ID {child_id})', color='red', zorder=5)

    plt.plot(predicted_growth_data['Age'], predicted_growth_data['Height'], label='Predicted Growth Data', color='green')

    plt.xlabel('Age')
    plt.ylabel('Height')
    plt.legend()
    plt.title('User Data, Predicted Growth Data, and Closest Matching Data')
    plt.show()

    return predicted_growth_data


# Usage:
user_data = [(8, 130), (10, 140), (12, 150), (14, 160), (16, 170)]
predicted_growth_data = predict_and_plot_growth_data(user_data, melted_data, kmeans, growth_curve)
print(predicted_growth_data)
z=z

def predict_and_plot_growth_data(user_data, melted_data, unmelted_data, kmeans, growth_curve):
    predicted_growth_data = pd.DataFrame(columns=['Age', 'Height'])
    closest_matching_indices = []
    for user_age, user_height in user_data:
        # Calculate distances from user-provided data point to all points in the melted data
        distances = cdist(np.array([[user_age, user_height]]), melted_data[['Age', 'Height']], metric='euclidean')[0]
        print(distances)
        closest_data_index = np.argmin(distances)

        # Find the participant ID for the closest data point
        participant_id = melted_data.iloc[closest_data_index]['child_id']

        # Get the corresponding cluster label for the participant from the unmelted data
        participant_row = unmelted_data[unmelted_data['child_id'] == participant_id]
        if not participant_row.empty:
            closest_cluster_label = participant_row['ClusterLabel'].iloc[0]
            closest_cluster_features = kmeans.cluster_centers_[closest_cluster_label]
            print(closest_cluster_features)
            # Predict heights for future ages using the growth curve
            ages = np.arange(user_age, 19)
            heights = growth_curve(ages, *closest_cluster_features[:3])
            print(heights)
            # Append the predicted data to the DataFrame
            predicted_growth_data = pd.concat(
                [predicted_growth_data, pd.DataFrame({'Age': ages, 'Height': heights})],
                ignore_index=True
            )
        else:
            # Handle the case where participant_id is not found in unmelted_data
            continue
        # Plot the user data, predicted data, and closest matching data
    plt.figure(figsize=(10, 6))
    plt.scatter(melted_data['Age'], melted_data['Height'], label='Closest Matching Data', color='blue')
    plt.scatter([user_age for user_age, _ in user_data], [user_height for _, user_height in user_data], label='User Data', color='red')
    plt.plot(predicted_growth_data['Age'], predicted_growth_data['Height'], label='Predicted Growth Data', color='green')
    plt.xlabel('Age')
    plt.ylabel('Height')
    plt.legend()
    plt.title('User Data, Predicted Growth Data, and Closest Matching Data')
    #plt.show()

    return predicted_growth_data

# Usage:
user_data = [(8, 130), (10, 140), (12, 150), (14, 160), (16, 170)]
predicted_growth_data = predict_and_plot_growth_data(user_data, melted_data, features_df, kmeans, growth_curve)
print(predicted_growth_data)
z=z



user_data = [(8, 130), (10, 140), (12, 150), (14, 160), (16, 170)]
predicted_growth_data = predict_and_plot_growth_data(user_data, melted_data, kmeans, growth_curve)
print(predicted_growth_data)
def find_closest_cluster(user_data, data, features_df, kmeans, n_closest=100):
    # Step 1: Match user input data with the closest examples from the training data
    # Initialize closest_examples as an empty DataFrame
    closest_examples = pd.DataFrame(columns=data.columns)
    matched_data = pd.DataFrame(columns=data.columns)  # To store matched training data

    for age, height in user_data:
        # Compute the absolute difference with all data points
        data['diff'] = abs(data['Height'] - height) + abs(data['Age'] - age)
        # Sort by difference and take the top N closest examples
        closest_data = data.nsmallest(n_closest, 'diff')
        closest_examples = pd.concat([closest_examples, closest_data], ignore_index=True)

        # Track matched training data
        matched_data = pd.concat([matched_data, closest_data], ignore_index=True)
    print(closest_examples)
    print('next')
    
    # Step 2: Identify the most common cluster from these matches
    # Get the cluster labels for the closest examples
    closest_examples['cluster'] = kmeans.predict(closest_examples[parameter_names])
    
    # Find the most common cluster label
    most_common_cluster = closest_examples['cluster'].mode()[0]

    # Step 3: Use the features from the chosen cluster to create a growth curve
    # Get the cluster center for the most common cluster
    cluster_center = kmeans.cluster_centers_[most_common_cluster]
    
    # Define ages for the growth curve (assuming ages 0-18)
    ages = np.arange(0, 19)
    
    # Calculate the growth curve based on the cluster center
    growth_curve_values = growth_curve(ages, *cluster_center[:3])
    
    # Step 4: Calculate parameters for the generated curve
    # Fit the curve to get the parameters
    params, _ = curve_fit(growth_curve, ages, growth_curve_values, p0=cluster_center[:3])
    
    # Prepare the matched training data for plotting
    matched_ages = matched_data['Age'].values
    matched_heights = matched_data['Height'].values

    # Plot the predicted curve and matched training data
    plt.figure(figsize=(10, 6))
    plt.scatter(matched_ages, matched_heights, label='Matched Training Data', color='blue')
    plt.plot(ages, growth_curve(ages, *params), label='Predicted Growth Curve', color='red')
    plt.xlabel('Age')
    plt.ylabel('Height')
    plt.legend()
    plt.title('Predicted Growth Curve vs. Matched Training Data')
    plt.show()
    
    return params

# Usage:
curve_params = find_closest_cluster(user_data, melted_data, features_df, kmeans)


z=z








def find_closest_cluster(user_data, scaler, kmeans_model, growth_curve_function):
    # Assuming user_data is a list of tuples like (age, height)

    # Normalize the user input
    user_data_normalized = scaler.transform(user_data)

    # Find the closest cluster
    closest_cluster = None
    smallest_distance = float('inf')
    for i, center in enumerate(kmeans_model.cluster_centers_):
        distance = np.linalg.norm(user_data_normalized - center)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_cluster = i

    # Generate a growth curve based on the closest cluster centroid
    centroid = kmeans_model.cluster_centers_[closest_cluster]
    age_range = np.arange(user_data[0][0], user_data[-1][0] + 1)  # Assuming user_data is sorted by age
    generated_curve = growth_curve_function(age_range, *centroid)

    # Fit the growth curve to the user data to adjust the centroid-based curve to the individual
    params, _ = curve_fit(growth_curve_function, [age for age, _ in user_data], [height for _, height in user_data], p0=centroid)

    return params, generated_curve
z=z



# Function to calculate the distance between two sets of parameters
def parameter_distance(params1, params2):
    return np.linalg.norm(params1 - params2)
# Initialize variables to keep track of the closest cluster
closest_cluster_label = None
min_distance = float('inf')
cluster_centers = kmeans.cluster_centers_

print(cluster_centers)
z=z
# Iterate through each cluster center
for cluster_label, cluster_center in enumerate(cluster_centers):
    # Fit the growth curve to the user-provided data
    try:
        params, _ = curve_fit(growth_curve, [age for age, _ in user_data], [height for _, height in user_data])
    except RuntimeError:
        # Handle fitting errors if necessary
        pass

    # Calculate the distance between user-provided parameters and cluster center
    distance = parameter_distance(params, cluster_center)

    # Check if this cluster is closer than the previous closest
    if distance < min_distance:
        min_distance = distance
        closest_cluster_label = cluster_label

# Use the parameters from the closest cluster to predict final height at age 18
final_height_at_18 = growth_curve(18, cluster_centers[closest_cluster_label][0], cluster_centers[closest_cluster_label][1], cluster_centers[closest_cluster_label][2])

print(f'Predicted final height at age 18: {final_height_at_18} cm')
z=z



#This will fit a growth curve to each age and height pair individually, and then calculate the distances between the resulting parameters and cluster centroids.
distances = []
# Create an array to store predicted heights
predicted_heights = []

def calculate_growth_curve(age, params):
    a, b, c = params
    return a / (1 + np.exp(b - c * age))
# Iterate through age and height pairs provided by the user
for age, height in user_data:
    try:
        # Fit a growth curve for the provided age and height pair
        params, _ = curve_fit(calculate_growth_curve, age, height)
        
        # Calculate distances between the parameters and cluster centroids
        cluster_distances = [np.linalg.norm(params - centroid) for centroid in kmeans.cluster_centers_]
        
        # Select the cluster with the smallest distance as the best match
        best_cluster = np.argmin(cluster_distances)
        
        # Use the parameters of the best matching cluster to predict the growth curve to age 18
        predicted_curve = calculate_growth_curve(np.arange(age, 19), *kmeans.cluster_centers_[best_cluster])
        
        # Append the predicted heights to the array
        predicted_heights.append(predicted_curve)
    except RuntimeError:
        # Handle fitting errors if necessary
        pass


z=z




# Step 2: Find the best matching cluster label for each user data point
best_matching_labels = np.argmin(distances, axis=1)

# Step 3: Aggregate Parameters for Each Cluster
cluster_params = {}
for label, (age, height) in zip(best_matching_labels, user_data):
    if label not in cluster_params:
        cluster_params[label] = []
    # Fit a growth curve to each age-height pair
    try:
        params, _ = curve_fit(growth_curve, age, height)
        cluster_params[label].append(params)
    except RuntimeError:
        # Handle fitting errors if necessary
        pass

# Calculate aggregated parameters for each cluster
aggregated_params = {}
for label, params_list in cluster_params.items():
    aggregated_params[label] = np.mean(params_list, axis=0)

# Step 4: Determine the Best Matching Cluster for Each User Data Point
predicted_ages = []
predicted_heights = []

for label, params in zip(best_matching_labels, user_data):
    best_matching_params = aggregated_params[label]
    
    predicted_age = 18
    predicted_height = growth_curve(predicted_age, *best_matching_params)
    
    predicted_ages.append(predicted_age)
    predicted_heights.append(predicted_height)

# Print the predicted ages and heights for each user data point
for i, (age, height) in enumerate(user_data):
    print(f"User {i + 1} - Age: {age}, Height: {height}, Predicted Age: {predicted_ages[i]}, Predicted Height: {predicted_heights[i]}")


z=z
# Function to calculate parameters from user input
def calculate_parameters(age_data, height_data):
    try:
        params, _ = curve_fit(growth_curve, age_data, height_data, maxfev=5000)
    except RuntimeError as e:
        print(f"Could not fit curve for user data: {e}")
        params = [0, 0, 0]
    return params

# Function to predict height at age 18 based on user input
def predict_height(user_input, scaler, clustering_model):
    age_data, height_data = user_input

    # Calculate parameters and scale
    params = calculate_parameters(age_data, height_data)
    scaled_params = scaler.transform([params])

    # Predict cluster for user data
    cluster = clustering_model.predict(scaled_params)

    # Filter the dataset for the predicted cluster
    cluster_data = features_df[kmeans.labels_ == cluster[0]]

    # Calculate predicted height at age 18 for each child in the cluster
    predicted_heights = []
    for _, row in cluster_data.iterrows():
        params_cluster = [row['PHV_Age'], row['PHV_Value'], row['Acceleration'], row['Curve_Param_a'], row['Curve_Param_b'], row['Curve_Param_c']]
        predicted_height = growth_curve(18, *params_cluster)
        predicted_heights.append(predicted_height)

    # Return the range of predicted heights
    return predicted_heights

# Example user input
user_input = ([8, 10, 11, 12, 13], [130, 135, 140, 150, 160])

# Predict height at age 18
predicted_heights = predict_height(user_input, scaler, kmeans)

# Print the range of predicted heights
print("Predicted Height Range at Age 18:", min(predicted_heights), "cm to", max(predicted_heights), "cm")

