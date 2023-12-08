import os
from scipy import stats
os.chdir('src')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Assuming you have a DataFrame `data` with individual growth data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')


def plot_growth_curve_and_save(data, child_id, directory="curve_images"):
    # Ensure the target directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    ages = data['Age']
    heights = data['Height']
    
    # Interpolate the curve for a smooth plot
    f = interp1d(ages, heights, kind='quadratic')
    age_range = np.linspace(min(ages), max(ages), num=100)
    height_range = f(age_range)
    
    # Plot the curve
    plt.figure()
    plt.plot(age_range, height_range)
    plt.scatter(ages, heights)  # Plot the actual data points
    plt.title(f"Growth Curve for Child ID: {child_id}")
    plt.xlabel('Age')
    plt.ylabel('Height (cm)')
    
    # Save the plot as an image
    plt.savefig(f"{directory}/{child_id}.png")
    plt.close()

# Example usage for one individual
plot_growth_curve_and_save(individual_data, 'child_1')
