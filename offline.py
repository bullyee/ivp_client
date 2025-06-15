import cv2
import os
import numpy as np
import pickle
import pandas as pd
from scipy.ndimage import histogram
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from KeypointData import KeypointData, ImageData, DbData
import ast

# Paths for saving models
data_folder = "saved_frames_home"  # Folder containing images
output_folder = "outcome_home"  # Folder to save outcomes
os.makedirs(output_folder, exist_ok=True)

kmeans_path = os.path.join(output_folder, "kmeans.pkl")
kdtree_path = os.path.join(output_folder, "kdtree.pkl")
dbdata_path = os.path.join(output_folder, "dbdata.pkl")

# Parameters
num_clusters = 500  # Number of visual words
threshold = 250.0  # Matching distance threshold


def read_metadata(file_path):
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


def read_image_data(folder):
    image_data_list = []

    for i in range(0, 31):
        keypoints_path = os.path.join(folder, f'frame_{i}_keypoints.pkl')
        metadata_path = os.path.join(folder, f'frame_{i}_metadata.csv')

        if os.path.exists(keypoints_path) and os.path.exists(metadata_path):
            with open(keypoints_path, "rb") as f:
                keypoint_data = pickle.load(f)
            R, T = read_metadata(metadata_path)
            if keypoint_data and isinstance(keypoint_data, list) and len(keypoint_data) > 0:
                image_data_list.append(ImageData(keypoint_data, R, T))
            else:
                print(f"Warning: No keypoints in {keypoints_path}")

    return image_data_list

def create_vocabulary(images, num_clusters):
    """Clusters all keypoint descriptors using KMeans and stores visual words in a K-d tree."""
    descriptors = []

    for image in images:
        if isinstance(image.keypointData, list) and len(image.keypointData) > 0:
            descriptors.extend([kp.descriptor for kp in image.keypointData])

    if not descriptors:
        raise ValueError("No descriptors found in the dataset.")

    descriptors = np.array(descriptors)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    kmeans.fit(descriptors)

    vocabulary = kmeans.cluster_centers_
    kdtree = cKDTree(vocabulary)

    return vocabulary, kdtree


def match_image_to_vocabulary(image_data, kdtree, num_clusters, threshold=300.0):
    """Matches an image's ORB descriptors to the visual words in the vocabulary."""
    word_counts = np.zeros(num_clusters, dtype=int)

    if not image_data.keypointData:
        return word_counts  # No keypoints, return empty histogram

    descriptors = np.array([kp.descriptor for kp in image_data.keypointData])
    distances, indices = kdtree.query(descriptors)

    for distance, index in zip(distances, indices):
        if distance < threshold:
            word_counts[index] += 1

    return word_counts



def process_database():
    imagesDatas = read_image_data(data_folder)
    print(f"Loaded {len(imagesDatas)} images")

    vocab, kdtree = create_vocabulary(imagesDatas, num_clusters)
    print(f"Vocabulary Shape: {vocab.shape}")

    histograms = []
    for image in imagesDatas:
        histogram = match_image_to_vocabulary(image, kdtree, num_clusters, threshold)
        if histogram is not None and np.sum(histogram) > 0:
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(num_clusters))  # Empty image gets a zero histogram
    # Plot the first histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_clusters), histograms[5], color='blue')
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("BoVW Histogram for First Image")
    plt.show()

    DbDatas = []
    for i in range(0, len(imagesDatas)):
        kpdata_global_coords = []
        for kp in imagesDatas[i].keypointData:
            global_coord = np.dot(imagesDatas[i].R, kp.world_coord.reshape(3,1)) + imagesDatas[i].T
            kpdata_global_coords.append(KeypointData(kp.pixel, kp.descriptor, global_coord.flatten()))
        DbDatas.append(DbData(kpdata_global_coords, histograms[i]))
    with open(dbdata_path, "wb") as f:
        pickle.dump(DbDatas, f)
    with open(kmeans_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(kdtree_path, "wb") as f:
        pickle.dump(kdtree, f)




    print("Database processed and saved in the outcome folder!")


process_database()
