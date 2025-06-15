import cv2
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Paths
frame_num = 31  # Change this to visualize a different frame
dbdata_path = "data_outcome/dbdata.pkl"
image_path = f"data/frame_{frame_num}_color.png"  # Change index as needed

# Load the database
with open(dbdata_path, "rb") as f:
    db_data_list = pickle.load(f)

# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Get image data from the database
if frame_num >= len(db_data_list):
    print(f"Error: Frame {frame_num} not found in the database.")
    exit()

image_data = db_data_list[frame_num]

# Draw keypoints
for kp in image_data.keypointData:
    u, v = kp.pixel  # Get pixel coordinates
    cv2.circle(image, (u, v), 3, (0, 255, 0), -1)  # Draw small green circle

# Select a few keypoints randomly to annotate
num_annotations = min(5, len(image_data.keypointData))  # Choose up to 5 keypoints
selected_keypoints = random.sample(image_data.keypointData, num_annotations)

# Draw text annotations for selected keypoints
for kp in selected_keypoints:
    u, v = kp.pixel  # Pixel coordinates
    x, y, z = kp.world_coord.flatten()  # Global coordinates (flatten to 1D)

    # Draw text (global coordinates)
    text = f"({x:.2f}, {y:.2f}, {z:.2f})"
    cv2.putText(image, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Display image with keypoints and annotations
cv2.imshow("Image with Keypoints and Global Coords", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the histogram
histogram = image_data.histogram
print(f"Histogram for frame {frame_num}:")
print(histogram)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(range(len(histogram)), histogram, color='blue')
plt.xlabel("Visual Word Index")
plt.ylabel("Frequency")
plt.title(f"BoVW Histogram for Frame {frame_num}")
plt.show()
