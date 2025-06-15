import cv2
import os
import numpy as np
import pickle
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from KeypointData import KeypointData, ImageData, DbData
import ast

"""
BoVW offline builder – ORB / AKAZE friendly
==========================================
Prints the descriptor count per image and the grand total before K‑Means runs.
No other logic changed.
"""

# ─── Paths ──────────────────────────────────────────────────────────────────
data_folder   = "data"          # where *_keypoints.pkl live
output_folder = "data_outcome"  # outputs
os.makedirs(output_folder, exist_ok=True)

kmeans_path = os.path.join(output_folder, "kmeans.pkl")
kdtree_path = os.path.join(output_folder, "kdtree.pkl")
dbdata_path = os.path.join(output_folder, "dbdata.pkl")

# ─── Parameters ─────────────────────────────────────────────────────────────
num_clusters = 500
threshold    = 400.0

# ─── Helpers ────────────────────────────────────────────────────────────────

def read_metadata(csv_path):
    meta = pd.read_csv(csv_path, header=None, names=["key", "value"])
    meta["key"] = meta["key"].str.strip()
    R = np.array(ast.literal_eval(meta.loc[meta.key == "R", "value"].values[0]))
    T = np.array(ast.literal_eval(meta.loc[meta.key == "T", "value"].values[0])).reshape(3, 1)
    return R, T


def read_image_data(folder):
    image_data_list = []

    for i in range(0, 35):
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

# ─── BoVW core ──────────────────────────────────────────────────────────────

def create_vocabulary(images, k):
    """Collect descriptors → float32 → K‑Means. Prints per‑image counts."""
    descriptor_rows = []
    total = 0
    for idx, img in enumerate(images):
        cnt = 0
        for kp in img.keypointData:
            if kp.descriptor is not None:
                descriptor_rows.append(kp.descriptor.astype(np.float32))
                cnt += 1
        print(f"image {idx}: descriptors = {cnt}")
        total += cnt
    print("Total descriptors across all images:", total)

    if total == 0:
        raise ValueError("No descriptors found in dataset – check input files.")

    descriptors = np.vstack(descriptor_rows)  # shape (N, D)

    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
    kmeans.fit(descriptors)

    vocab  = kmeans.cluster_centers_
    kdtree = cKDTree(vocab)
    return vocab, kdtree


def match_image_to_vocabulary(img, kdtree, k, thr=300.0):
    hist = np.zeros(k, dtype=int)
    if img.keypointData:
        desc = np.vstack([kp.descriptor for kp in img.keypointData]).astype(np.float32)
        dists, idx = kdtree.query(desc)
        hist += np.bincount(idx[dists < thr], minlength=k)
    return hist

# ─── Main ───────────────────────────────────────────────────────────────────

def process_database():
    images = read_image_data(data_folder)
    print("Loaded", len(images), "images")

    vocab, kdtree = create_vocabulary(images, num_clusters)
    print("Vocabulary shape:", vocab.shape)

    hists = [match_image_to_vocabulary(img, kdtree, num_clusters, threshold) for img in images]

    if len(hists) > 5:
        plt.figure(figsize=(10, 5))
        plt.bar(range(num_clusters), hists[5])
        plt.title("BoVW histogram – image 5")
        plt.xlabel("Visual word")
        plt.ylabel("Freq")
        plt.show()

    db_entries = []
    for img, hist in zip(images, hists):
        kp_global = [KeypointData(kp.pixel, kp.descriptor,
                                  (img.R @ kp.world_coord.reshape(3, 1) + img.T).flatten())
                     for kp in img.keypointData]
        db_entries.append(DbData(kp_global, hist))

    with open(dbdata_path,  "wb") as f: pickle.dump(db_entries, f)
    with open(kmeans_path, "wb") as f: pickle.dump(vocab,      f)
    with open(kdtree_path, "wb") as f: pickle.dump(kdtree,     f)

    print("✅  Database ready – files stored in", output_folder)


if __name__ == "__main__":
    process_database()
