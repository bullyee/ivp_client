import cv2
import os
import pickle
import numpy as np
import pandas as pd
import ast

# Paths
image_folder = "data"

# Sorting available images
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith("_color.png")])
num_images = len(image_files)
current_index = 0


def load_keypoints(file_path):
    """Load keypoints (camera coordinates) from pkl file."""
    with open(file_path, "rb") as f:
        keypoints = pickle.load(f)
    return keypoints  # Returns a list of KeypointData objects


def load_metadata(file_path):
    """Reads metadata CSV file and extracts R and T matrices correctly."""
    # Read CSV as a normal table (not using index_col=0)
    metadata = pd.read_csv(file_path, delimiter=",", header=None)

    # Ensure column 0 contains R/T names, and column 1 contains values
    metadata.columns = ["key", "value"]

    # Strip spaces and remove any unwanted characters
    metadata["key"] = metadata["key"].str.strip()

    # Convert R and T safely
    R_str = metadata.loc[metadata["key"] == "R", "value"].values[0].strip()
    T_str = metadata.loc[metadata["key"] == "T", "value"].values[0].strip()

    R = np.array(ast.literal_eval(R_str)) if R_str else np.eye(3)
    T = np.array(ast.literal_eval(T_str)).reshape((3, 1)) if T_str else np.zeros((3, 1))

    return R, T


def camera_to_global(camera_coord, R, T):
    """Convert a 3D point from camera coordinates to global coordinates."""
    global_coord = np.dot(R, camera_coord.reshape(3, 1)) + T
    return global_coord

def draw_keypoints(image, keypoints, R=None, T=None):
    """
    Draws keypoints and labels some with camera/global coordinates.
    If R and T are provided, converts to global coordinates.
    """
    for i, kp in enumerate(keypoints):
        u, v = kp.pixel  # Extract pixel coordinates
        coord = kp.world_coord.flatten()

        # If R and T are given, convert to global coordinates
        if R is not None and T is not None:
            coord = camera_to_global(coord, R, T).flatten()

        x, y, z = coord

        # Draw the keypoint
        cv2.circle(image, (u, v), 4, (0, 255, 0), -1)

        # Label every 10th keypoint with its coordinates
        if i % 100 == 0:
            text = f"({x:.2f}, {y:.2f}, {z:.2f})"
            cv2.putText(image, text, (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return image


def show_image(index):
    """
    Display the selected image with keypoints in camera and global coordinates.
    """
    if index < 0 or index >= num_images:
        return

    # Paths for color image, keypoints, and metadata
    image_path = os.path.join(image_folder, image_files[index])
    keypoints_path = image_path.replace("_color.png", "_keypoints.pkl")
    metadata_path = image_path.replace("_color.png", "_metadata.csv")

    if not os.path.exists(keypoints_path) or not os.path.exists(metadata_path):
        print(f"Missing keypoints or metadata file for {image_path}")
        return

    # Load image, keypoints, and metadata
    image = cv2.imread(image_path)
    keypoints = load_keypoints(keypoints_path)
    R, T = load_metadata(metadata_path)

    # Create two copies for displaying side by side
    image_camera = image.copy()
    image_global = image.copy()

    # Draw keypoints in camera and global coordinates
    image_camera = draw_keypoints(image_camera, keypoints)
    image_global = draw_keypoints(image_global, keypoints, R, T)

    # Stack images side by side
    combined_image = np.hstack((image_camera, image_global))

    # Display the combined image
    cv2.imshow("Camera Coord (Left) | Global Coord (Right)", combined_image)


# Start Viewer
cv2.namedWindow("Camera Coord (Left) | Global Coord (Right)", cv2.WINDOW_NORMAL)
show_image(current_index)
key = cv2.waitKey(0) & 0xFF


cv2.destroyAllWindows()
