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
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=24,
                                  inter_op_parallelism_threads=24,
                                  allow_soft_placement=True,
                                  device_count={'CPU': 1})

# Create a session with the above configuration
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# Load your data
data = pd.read_csv('svk_height_weight_mens_2008_v2.csv')

# Convert height measurements from mm to cm
height_columns = [col for col in data.columns if col.startswith('ATV_')]
data[height_columns] = data[height_columns] / 10
z_scores = stats.zscore(data[height_columns])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]
# Reshape the data into participant-age-height format
participant_data = pd.melt(data, id_vars=['child_id'], value_vars=height_columns, var_name='Age', value_name='Height')

# Drop rows with missing height values
participant_data = participant_data.dropna(subset=['Height'])

# Extract ages and heights
ages = participant_data['Age'].str.extract('(\d+)').astype(int)
heights = participant_data['Height']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(ages, heights, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values.reshape(-1, 1))
from joblib import dump, load

# After fitting the scaler, save it
scaler = scaler.fit(X_train) 
dump(scaler, 'scaler.joblib')


X_test_scaled = scaler.transform(X_test.values.reshape(-1, 1))

# Neural network architecture
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.1)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test)
print(f'Test loss: {test_loss}')

# Predict height at age 18 using the model
age_18_scaled = scaler.transform(np.array([[18]]))
predicted_height_at_18 = model.predict(age_18_scaled)
print(f'Predicted height at age 18: {predicted_height_at_18[0][0]} cm')

model.save('my_model.keras')




z=z
# Later on, load the model (for example, in another script or in the GUI)
from tensorflow.keras.models import load_model

loaded_model = load_model('my_model.h5')