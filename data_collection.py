import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pickle
import pandas as pd
from KeypointData import KeypointData

assert cv2.__version__.startswith("4")

# Output directory
output_dir = "data1"
os.makedirs(output_dir, exist_ok=True)

def depth_to_3d(u, v, depth, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth  # Depth is already in meters
    return np.array([X, Y, Z])

def apply_clahe(image):
    # Convert to grayscale (CLAHE works on single channel images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a CLAHE object with a clip limit and tile grid size (you can adjust these parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized

def extract_orb_keypoints_with_world_coords(color_image, depth_image, K):
    """Extracts ORB keypoints and computes their world coordinates."""
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(color_image, None)
    keypoints_info = []

    for kp, desc in zip(keypoints, descriptors):
        u, v = int(kp.pt[0]), int(kp.pt[1])
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_image[v, u] * depth_scale # Convert mm to meters


        if depth_m > 0:  # Ignore invalid depth values
            world_coord = depth_to_3d(u, v, depth_m, K)
            keypoints_info.append(KeypointData((u, v), desc, world_coord))

    return keypoints_info

def extract_akaze_keypoints_with_world_coords(color_image, depth_image, K):
    """Extracts ORB keypoints and computes their world coordinates."""
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(color_image, None)
    keypoints_info = []

    if descriptors is None:
        keypoints_info = []
    else:
        for kp, desc in zip(keypoints, descriptors):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_m = depth_image[v, u] * depth_scale  # Convert mm to meters

            if depth_m > 0:  # Ignore invalid depth values
                world_coord = depth_to_3d(u, v, depth_m, K)
                keypoints_info.append(KeypointData((u, v), desc, world_coord))

    return keypoints_info


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get intrinsics for color and depth sensors
depth_sensor = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_sensor = profile.get_stream(rs.stream.color).as_video_stream_profile()

depth_intrinsics = depth_sensor.get_intrinsics()
color_intrinsics = color_sensor.get_intrinsics()

# Print intrinsics
print("Depth Camera Intrinsics:")
print(f"  Width: {depth_intrinsics.width}, Height: {depth_intrinsics.height}")
print(f"  Focal Length (fx, fy): {depth_intrinsics.fx}, {depth_intrinsics.fy}")
print(f"  Principal Point (ppx, ppy): {depth_intrinsics.ppx}, {depth_intrinsics.ppy}")

print("\nColor Camera Intrinsics:")
print(f"  Width: {color_intrinsics.width}, Height: {color_intrinsics.height}")
print(f"  Focal Length (fx, fy): {color_intrinsics.fx}, {color_intrinsics.fy}")
print(f"  Principal Point (ppx, ppy): {color_intrinsics.ppx}, {color_intrinsics.ppy}")

K = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
              [0, color_intrinsics.fy, color_intrinsics.ppy],
              [0, 0, 1]])

print("\nCamera Intrinsic Matrix:\n", K)
print("\nCamera Started")

frame_count = 0

try:
    align = rs.align(rs.stream.color)
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Convert depth to colormap for display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv2.imshow('Aligned Depth and Color', images)

        # Save images and extract keypoints when 's' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            color_path = os.path.join(output_dir, f"frame_{frame_count}_color.png")
            keypoints_path = os.path.join(output_dir, f"frame_{frame_count}_keypoints.pkl")
            metadata_path = os.path.join(output_dir, f"frame_{frame_count}_metadata.csv")

            cv2.imwrite(color_path, color_image)

            norm_image = apply_clahe(color_image)
            keypoints_data = extract_akaze_keypoints_with_world_coords(norm_image, aligned_depth_image, K)
            with open(keypoints_path, "wb") as f:
                pickle.dump(keypoints_data, f)

            # Placeholder for R and T matrices (identity and zero translation for now)
            R = np.eye(3).tolist()
            T = np.zeros((3, 1)).tolist()

            metadata = pd.DataFrame({"R": [R], "T": [T]}).T
            #metadata.to_csv(metadata_path, index=False)

            print(f"Saved frame {frame_count} and extracted {len(keypoints_data)} keypoints.")
            frame_count += 1

        # Quit when 'q' is pressed
        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
