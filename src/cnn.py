import os
from scipy import stats
os.chdir('src')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import interp1d

# Load your data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Convert height measurements from mm to cm
height_columns = [col for col in data.columns if col.startswith('ATV_')]
data[height_columns] = data[height_columns] / 10

# Define a function to plot a growth curve
def plot_and_save_growth_curve(data_row, directory="growth_curve_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create a figure
    plt.figure(figsize=(5, 5))
    ages = np.arange(8, 19)
    heights = data_row[height_columns].values
    # Interpolation for a smoother curve
    cubic_interp = interp1d(ages, heights, kind='cubic')
    ages_smooth = np.linspace(ages.min(), ages.max(), 200)
    heights_smooth = cubic_interp(ages_smooth)
    
    plt.plot(ages_smooth, heights_smooth, 'b-', linewidth=2)
    plt.scatter(ages, heights, edgecolor='r', facecolor='white', s=50)
    
    plt.xlabel('Age')
    plt.ylabel('Height (cm)')
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig(f"{directory}/{data_row['child_id']}.png")
    plt.close()

# Generate and save growth curve images for all individuals
for index, row in data.iterrows():
    plot_and_save_growth_curve(row)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define a simple CNN architecture
def create_cnn(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))  # For regression output
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
