import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Read the Arduino data
data = pd.read_csv('simulated_arduino_A1.csv')

# Create figure with transparent background
fig = plt.figure(figsize=(10, 6), facecolor='none')
ax = fig.add_subplot(111)

# Normalize and amplify the variations in the data
# Subtract the mean and divide by standard deviation to highlight variations
mean_val = data['A1'].mean()
std_val = data['A1'].std()
# Apply normalization and amplify by a factor
amplification_factor = 5
normalized_data = (data['A1'] - mean_val) / std_val * amplification_factor

# Optional: Apply some smoothing to make trends more visible
window_size = 30
smoothed_data = signal.savgol_filter(normalized_data, window_size, 3) if len(normalized_data) > window_size else normalized_data

# Plot the amplified data with a black line
ax.plot(data['Time(ms)'], smoothed_data, color='black', linewidth=1.5)

# Remove all axes, grid, and frame
ax.set_axis_off()

# Set background to transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Save the visualization with transparent background
plt.savefig('arduino_visualization.png', 
            transparent=True, 
            bbox_inches='tight',
            pad_inches=0)

print("Visualization created and saved as 'arduino_visualization.png'") 