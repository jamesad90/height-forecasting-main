import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import os
os.chdir('src')
import pandas as pd
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=24,
                                  inter_op_parallelism_threads=24,
                                  allow_soft_placement=True,
                                  device_count={'CPU': 1})
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

print(melted_data)

df = melted_data
X = df['Age'].values.reshape(-1, 1)  # reshaping for compatibility with Keras
y = df['Height'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture with regularization
model = Sequential([
    Dense(64, input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Dropout for regularization
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10, verbose=1)

# Evaluate the model on the validation set
val_loss = model.evaluate(X_val, y_val, verbose=0)

# Predict growth values (heights) for new ages using the trained model
predicted_heights = model.predict(X_val)

# Plot training history for loss and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()