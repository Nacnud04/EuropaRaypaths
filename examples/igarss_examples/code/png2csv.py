import cv2
import numpy as np
import pandas as pd

# Load image
image = cv2.imread('figures/provided_ridge.png')
height, width, _ = image.shape

# Convert to HSV for better color filtering
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define blue color range in HSV
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Create mask for blue line
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find coordinates of blue pixels
coords = np.column_stack(np.where(mask > 0))

# Estimate axis ranges from your description
x_pixel_min, x_pixel_max = 100, width - 100  # adjust if needed
y_pixel_min, y_pixel_max = height - 100, 100

x_data_min, x_data_max = -100, 100  # km
y_data_min, y_data_max = 0, 600     # m

# Convert pixel coordinates to data values
data_points = []
for y, x in coords:
    if x_pixel_min <= x <= x_pixel_max and y_pixel_max <= y <= y_pixel_min:
        x_val = ((x - x_pixel_min) / (x_pixel_max - x_pixel_min)) * (x_data_max - x_data_min) + x_data_min
        y_val = ((y_pixel_min - y) / (y_pixel_min - y_pixel_max)) * (y_data_max - y_data_min) + y_data_min
        data_points.append((x_val, y_val))

# Sort by x value
data_points.sort()

# Export to CSV
df = pd.DataFrame(data_points, columns=['Along Track (km)', 'Surface Height (m)'])

# Round x-values to desired precision (e.g., 0.1 km) to group similar values
df['Along Track (km)'] = df['Along Track (km)'].round(1)

# Group by x and average y
df_unique = df.groupby('Along Track (km)', as_index=False).mean()

# update x axis
df_unique['Along Track (km)'] = np.linspace(-30, 30, len(df_unique['Along Track (km)']))

# update y axis
df_unique['Surface Height (m)'] -= np.min(df_unique['Surface Height (m)'])

# Export to CSV
df_unique.to_csv('facets/dem_profile.csv', index=False)

print("Unique data extraction complete. CSV saved.")

import matplotlib.pyplot as plt

plt.plot(df_unique['Along Track (km)'], df_unique['Surface Height (m)'], linewidth=1)
plt.show()
