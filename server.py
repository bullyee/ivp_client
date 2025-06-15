import socket
import struct
import gzip
import json
import numpy as np
import cv2
import pickle
from typing import List
from KeypointData import DbData
import matplotlib.pyplot as plt

SERVER_IP = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 34566

# Load database data
with open("data_outcome/dbdata.pkl", "rb") as f:
    db_data : List[DbData] = pickle.load(f)  # List of DbData objects

with open("data_outcome/kdtree.pkl", "rb") as f:
    kdtree = pickle.load(f)  # Pre-trained K-d tree

with open("data_outcome/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)  # Pre-trained K-means model

# Parameters
num_clusters = 500  # Number of visual words
threshold = 400.0  # Matching distance threshold

# Start the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(5)
print(f"Listening on {SERVER_IP}:{SERVER_PORT}")

def recv_all(sock, length):
    """Receive exactly 'length' bytes from the socket."""
    data = b""
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError(f"Connection closed before receiving all bytes. Expected {length}, got {len(data)} bytes.")
        data += packet
    return data

def get_pixel(item):
    """Return the pixel coordinates from an item.
       If 'pixel' exists, use it; otherwise, use 'x' and 'y' keys."""
    if "pixel" in item:
        return item["pixel"]
    elif "x" in item and "y" in item:
        return [item["x"], item["y"]]
    else:
        raise KeyError("No pixel or x,y field found in item.")

def descriptors_to_histogram(descriptors: np.ndarray, kdtree, num_clusters: int, threshold: float = 300.0) -> np.ndarray:
    """
    Converts a set of ORB descriptors into a Bag-of-Visual-Words (BoVW) histogram
    using the provided k-d tree and clustering vocabulary.

    Parameters:
        descriptors (np.ndarray): ORB descriptors of shape (n_keypoints, 32)
        kdtree: Pre-trained k-d tree (e.g., from sklearn.neighbors.KDTree)
        num_clusters (int): Total number of visual words (clusters).
        threshold (float): Distance threshold to consider a descriptor as matched.

    Returns:
        np.ndarray: Histogram vector of size (num_clusters,) representing visual word frequency.
    """
    histogram = np.zeros(num_clusters, dtype=int)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    # Query the k-d tree for nearest visual word for each descriptor
    distances, indices = kdtree.query(descriptors, k=1)  # k=1 for closest visual word
    distances = distances.flatten()
    indices = indices.flatten()

    for dist, idx in zip(distances, indices):
        if dist < threshold:
            histogram[idx] += 1

    return histogram

def visualize_histogram(histogram, title="Visual Word Histogram"):
    """
    Visualizes a BoVW histogram as a bar chart.

    Parameters:
        histogram (np.ndarray): Array of visual word counts.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 4))
    x = np.arange(len(histogram))
    plt.bar(x, histogram, width=1.0, edgecolor='black')
    plt.title(title)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

# def retrieve_similar_image_tf_idf(query_histogram, db_data, num_clusters, top_k=3):
#     """
#     Computes TF-IDF for the query image and all database images, then retrieves the most similar image.
#
#     Parameters:
#         query_histogram: BoVW histogram of the query image.
#         db_data: List of DbData objects containing histograms.
#         num_clusters: Number of visual words.
#
#     Returns:
#         best_match: The most similar image from the database.
#     """
#
#     # Step 1: Compute IDF dynamically from the database
#     num_images = len(db_data)
#     histograms = np.array([db.histogram for db in db_data])
#
#     df = np.count_nonzero(histograms, axis=0)  # Document frequency
#     idf_vector = np.log((num_images + 1) / (df + 1)) + 1  # Smoothed IDF
#
#     # Step 2: Compute TF-IDF for the query image
#     tf_query = (query_histogram + 1) / (np.sum(query_histogram) + num_clusters)  # Smoothing TF
#     query_tf_idf = tf_query * idf_vector  # Apply IDF weighting
#
#     # Step 3: Compute TF-IDF for all database images
#     db_tf_idf_matrix = np.array([
#         (db.histogram + 1) / (np.sum(db.histogram) + num_clusters) * idf_vector
#         for db in db_data
#     ])
#
#     # Step 4: Compute similarity using dot product
#     similarities = np.dot(db_tf_idf_matrix, query_tf_idf)
#
#     # Step 5: Get the best match
#     top_matches_idx = np.argsort(similarities)[-top_k:][::-1]
#
#     print(f"Top {top_k} Matches: {top_matches_idx} | Similarity Scores: {similarities[top_matches_idx]}")
#
#     return top_matches_idx  # Return top-k indices


def retrieve_similar_image_tf_idf(query_histogram, db_data, num_clusters, top_k=3):
    """
    Retrieves the top-k similar images from the database using a TF-IDF weighting scheme
    as described in the paper.

    Parameters:
        query_histogram: 1D numpy array of counts for the query image (length = num_clusters).
        db_data: List of DbData objects; each DbData.histogram is a 1D numpy array of length num_clusters.
        num_clusters: Number of visual words.
        top_k: Number of top matches to return.

    Returns:
        top_matches_idx: numpy array of the top-k indices (in descending order of similarity).
    """
    num_images = len(db_data)
    # Create a matrix of histograms from the database.
    histograms = np.array([db.histogram for db in db_data])  # shape: (num_images, num_clusters)

    # Compute document frequency for each word (number of images in which the word appears).
    df = np.count_nonzero(histograms, axis=0).astype(np.float64)
    # Avoid division by zero; if a word never appears, set its df to 1.
    df[df == 0] = 1

    # Compute IDF for each visual word.
    idf_vector = np.log(num_images / df)  # Shape: (num_clusters,)

    # Compute TF for the query image.
    total_query = np.sum(query_histogram)
    if total_query == 0:
        total_query = 1  # avoid division by zero
    tf_query = query_histogram / total_query
    query_tf_idf = tf_query * idf_vector  # Shape: (num_clusters,)

    # Compute TF-IDF for each DB image.
    db_tf_idf_matrix = []
    for db in db_data:
        total_db = np.sum(db.histogram)
        if total_db == 0:
            total_db = 1
        tf_db = db.histogram / total_db
        db_tf_idf = tf_db * idf_vector
        db_tf_idf_matrix.append(db_tf_idf)
    db_tf_idf_matrix = np.array(db_tf_idf_matrix)  # Shape: (num_images, num_clusters)

    # Compute similarity using dot product.
    similarities = np.dot(db_tf_idf_matrix, query_tf_idf)

    # Retrieve indices for the top_k highest similarity scores.
    top_matches_idx = np.argsort(similarities)[-top_k:][::-1]
    print(f"Top {top_k} Matches: {top_matches_idx} | Similarity Scores: {similarities[top_matches_idx]}")

    return top_matches_idx


def find_best_matched_image(query_keypoints_xy, query_descriptors, top_matches, db_data):
    """
    Finds the best-matching image from the database using precomputed features and returns matching data.

    Parameters:
        query_keypoints_xy: NumPy array of shape (N, 2) with query keypoint coordinates.
        query_descriptors: NumPy array of shape (N, 32) with query descriptors (dtype=np.uint8).
        top_matches: List of indices (from retrieval) to consider from the database.
        db_data: List of DbData objects (each with keypointData having 'pixel' and 'descriptor').

    Returns:
        best_match_idx: The index of the best-matching DB image.
        best_inliers: List of match objects that are inliers after RANSAC.
        query_keypoints_xy: The original query keypoints (NumPy array).
        best_db_keypoints_xy: NumPy array of DB image keypoint coordinates for the best match.
    """

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    max_inliers = 0
    best_match_idx = None
    best_inliers = []
    best_db_keypoints_xy = None
    best_inlier_ratio = 0.0

    for match_idx in top_matches:
        data = db_data[match_idx]
        if not data.keypointData:
            continue

        # Use the precomputed keypoint coordinates and descriptors from db_data
        db_keypoints_xy = np.array([k.pixel for k in data.keypointData], dtype=np.float32)
        descriptors2 = np.array([k.descriptor for k in data.keypointData], dtype=np.uint8)

        if descriptors2 is None or len(descriptors2) == 0:
            continue

        # 1: Match descriptors between query and DB image
        matches = bf.knnMatch(query_descriptors, descriptors2, k=2)

        # 2: Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) < 4:
            continue


        # 3: Build coordinate arrays for homography computation
        src_pts = np.float32([query_keypoints_xy[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([db_keypoints_xy[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            continue

        num_inliers = np.sum(mask)
        inlier_ratio = num_inliers / len(good_matches)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_ratio = inlier_ratio
            best_match_idx = match_idx
            best_inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]
            best_db_keypoints_xy = db_keypoints_xy

    return best_match_idx, best_inliers, query_keypoints_xy, best_db_keypoints_xy, best_inlier_ratio

def compute_camera_pose(query_keypoints_xy, best_inliers, db_data_best, camera_matrix, dist_coeffs=None):
    """
    Computes the camera pose (rotation and translation) using EPnP given matched points.

    Parameters:
        query_keypoints_xy: NumPy array of shape (N, 2) containing query keypoint coordinates.
        best_inliers: List of match objects (from BFMatcher) that are inliers.
        db_data_best: The DbData object for the best matching DB image.
        camera_matrix: 3x3 intrinsic camera matrix.
        dist_coeffs: Distortion coefficients (default is None, meaning zeros).

    Returns:
        rvec: Rotation vector (Rodrigues).
        tvec: Translation vector.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # For each inlier, get the query image point.
    image_points = np.float32([query_keypoints_xy[m.queryIdx] for m in best_inliers]).reshape(-1, 2)

    # And get the corresponding 3D world point from the best DB image.
    object_points = []
    for m in best_inliers:
        # Access the trainIdx-th keypoint data from db_data_best.
        world_coord = db_data_best.keypointData[m.trainIdx].world_coord
        object_points.append(world_coord)
    object_points = np.float32(object_points)

    # EPnP via cv2.solvePnP.
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        print("EPnP failed.")
        return None, None
    return rvec, tvec


def compute_camera_pose_from_top_match(
        query_keypoints_xy, query_descriptors, best_match_index, db_data, camera_matrix,
        dist_coeffs=None, min_inliers=8, ransac_reproj_threshold=8.0, ransac_iters=100):
    """
    Robust camera pose estimation using EPnP + RANSAC.
    Handles domain gap and outlier correspondences.

    Returns:
        rvec, tvec, inliers (or None, None, None if failed)
    """

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    best_db_data = db_data[best_match_index]
    db_keypoints_xy = np.array([k.pixel for k in best_db_data.keypointData], dtype=np.float32)
    db_descriptors = np.array([k.descriptor for k in best_db_data.keypointData], dtype=np.uint8)
    db_world_coords = np.array([k.world_coord for k in best_db_data.keypointData], dtype=np.float32)

    # Descriptor matching (AKAZE/ORB: NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(query_descriptors, db_descriptors, k=2)

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good_matches) < 4:
        print("Not enough matches for EPnP (need at least 4).")
        return None, None, None

    # Correspondence arrays
    image_points = np.float32([query_keypoints_xy[m.queryIdx] for m in good_matches]).reshape(-1, 2)
    object_points = np.float32([db_world_coords[m.trainIdx] for m in good_matches]).reshape(-1, 3)

    # RANSAC-based EPnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=ransac_reproj_threshold,
        iterationsCount=ransac_iters
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        print(f"PnP RANSAC failed or too few inliers: {0 if inliers is None else len(inliers)}")
        return None, None

    print(f"[PnPRansac] pose found with {len(inliers)} inliers out of {len(good_matches)} matches.")
    return rvec, tvec

def bf_compute_camera_pose_from_top_matches(query_keypoints_xy, query_descriptors, match_indices, db_data, camera_matrix, dist_coeffs=None):
    """
    Computes camera pose (rotation and translation) using EPnP for a list of DB matches.

    Parameters:
        query_keypoints_xy: NumPy array of shape (N, 2) with query keypoint coordinates.
        query_descriptors: NumPy array of shape (N, 32) with query descriptors (dtype=np.uint8).
        match_indices: List of indices of the matching DB images.
        db_data: List of DbData objects.
        camera_matrix: 3x3 NumPy array with intrinsic camera parameters.
        dist_coeffs: Optional distortion coefficients.

    Returns:
        List of tuples: (rvec, tvec, match_idx, num_inliers)
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    results = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for match_idx in match_indices:
        best_db_data = db_data[match_idx]
        db_keypoints_xy = np.array([k.pixel for k in best_db_data.keypointData], dtype=np.float32)
        db_descriptors = np.array([k.descriptor for k in best_db_data.keypointData], dtype=np.uint8)
        db_world_coords = np.array([k.world_coord for k in best_db_data.keypointData], dtype=np.float32)

        # Match descriptors between query and DB image
        matches = bf.knnMatch(query_descriptors, db_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 4:
            print(f"[{match_idx}] Not enough matches for EPnP (need at least 4).")
            results.append((None, None, match_idx, 0))
            continue

        image_points = np.float32([query_keypoints_xy[m.queryIdx] for m in good_matches]).reshape(-1, 2)
        object_points = np.float32([db_world_coords[m.trainIdx] for m in good_matches]).reshape(-1, 3)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            print(f"[{match_idx}] EPnP failed.")
            results.append((None, None, match_idx, len(good_matches)))
        else:
            results.append((rvec, tvec, match_idx, len(good_matches)))

    return results

def bf_best_pose_from_top_matches(query_keypoints_xy, query_descriptors, match_indices, db_data, camera_matrix, dist_coeffs=None, top_k = 10):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    results = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    preselect = []

    for match_idx in match_indices:
        db_entry = db_data[match_idx]
        db_kp_xy = np.array([k.pixel for k in db_entry.keypointData], dtype=np.float32)
        db_des = np.array([k.descriptor for k in db_entry.keypointData], dtype=np.uint8)

        # Match descriptors between query and DB image
        matches = bf.knnMatch(query_descriptors, db_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        preselect.append((match_idx, good_matches))

    # Sort by good match count
    preselect.sort(key=lambda x: len(x[1]), reverse=True)
    top_candidates = preselect[:top_k]

    best_result = (None, None, None, 0, 0.0)

    for match_idx, good_matches in top_candidates:
        if len(good_matches) < 4:
            continue

        db_entry = db_data[match_idx]
        db_world = np.array([k.world_coord for k in db_entry.keypointData], dtype=np.float32)
        db_kp_xy = np.array([k.pixel for k in db_entry.keypointData], dtype=np.float32)

        img_pts = np.float32([query_keypoints_xy[m.queryIdx] for m in good_matches]).reshape(-1, 2)
        obj_pts = np.float32([db_world[m.trainIdx] for m in good_matches]).reshape(-1, 3)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            continue

        _, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP
        )

        inlier_count = len(inliers) if inliers is not None else 0
        inlier_ratio = inlier_count / len(good_matches)

        if inlier_ratio > best_result[4]:
            best_result = (rvec_r, tvec_r, match_idx, inlier_count, inlier_ratio)

    return best_result

def bf_compute_camera_pose_from_top_matches(query_keypoints_xy, query_descriptors, match_indices, db_data, camera_matrix, dist_coeffs=None):
    """
    Computes camera pose (rotation and translation) using EPnP for a list of DB matches.

    Parameters:
        query_keypoints_xy: NumPy array of shape (N, 2) with query keypoint coordinates.
        query_descriptors: NumPy array of shape (N, 32) with query descriptors (dtype=np.uint8).
        match_indices: List of indices of the matching DB images.
        db_data: List of DbData objects.
        camera_matrix: 3x3 NumPy array with intrinsic camera parameters.
        dist_coeffs: Optional distortion coefficients.

    Returns:
        List of tuples: (rvec, tvec, match_idx, num_inliers)
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    results = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for match_idx in match_indices:
        best_db_data = db_data[match_idx]
        db_keypoints_xy = np.array([k.pixel for k in best_db_data.keypointData], dtype=np.float32)
        db_descriptors = np.array([k.descriptor for k in best_db_data.keypointData], dtype=np.uint8)
        db_world_coords = np.array([k.world_coord for k in best_db_data.keypointData], dtype=np.float32)

        # Match descriptors between query and DB image
        matches = bf.knnMatch(query_descriptors, db_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 4:
            print(f"[{match_idx}] Not enough matches for EPnP (need at least 4).")
            results.append((None, None, match_idx, 0))
            continue

        image_points = np.float32([query_keypoints_xy[m.queryIdx] for m in good_matches]).reshape(-1, 2)
        object_points = np.float32([db_world_coords[m.trainIdx] for m in good_matches]).reshape(-1, 3)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            print(f"[{match_idx}] EPnP failed.")
            results.append((None, None, match_idx, len(good_matches)))
        else:
            results.append((rvec, tvec, match_idx, len(good_matches)))

    return results


def apply_clahe(image):
    # Convert to grayscale (CLAHE works on single channel images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a CLAHE object with a clip limit and tile grid size (you can adjust these parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr}")

    try:
        # Receive the length of the compressed data
        data_length = struct.unpack('!I', recv_all(client_socket, 4))[0]
        print(f"Expecting {data_length} bytes of compressed data")

        # Receive the compressed data
        compressed_data = recv_all(client_socket, data_length)
        print(f"Received {len(compressed_data)} bytes of compressed data")

        # Decompress and parse JSON payload
        json_data = gzip.decompress(compressed_data).decode("utf-8")
        payload = json.loads(json_data)

        # Check if payload is a dict with 'keypoints' key; if not, assume payload is the list of keypoints
        if isinstance(payload, dict) and "keypoints" in payload:
            keypoint_entries = payload["keypoints"]
            cam_mat = payload.get("cameraMatrix", None)
        else:
            keypoint_entries = payload
            cam_mat = None

        # Convert keypoint entries to keypoints_xy and descriptors.
        keypoints_xy = np.array([get_pixel(item) for item in keypoint_entries], dtype=np.float32)
        descriptors = np.array([item["descriptor"] for item in keypoint_entries], dtype=np.uint8)

        # if cam_mat is not None:
        #     camera_matrix = np.array(cam_mat, dtype=np.float64)
        #     print("Camera Matrix:")
        #     print(camera_matrix)
        # else:
        #     camera_matrix = None

        # print(f"Extracted {len(keypoints_xy)} keypoints and descriptors with shape {descriptors.shape}")

        # Extract camera matrix and convert to NumPy array (3x3)
        camera_matrix = np.array(payload["cameraMatrix"], dtype=np.float64)

        # print(f"Extracted {len(keypoints_xy)} keypoints and camera matrix:\n{camera_matrix}")

        # Generate histogram
        histogram = descriptors_to_histogram(descriptors, kdtree, num_clusters, threshold)
        #visualize_histogram(histogram, title="Query Image Histogram")

        # print("Descriptor count:", len(descriptors))
        # print("Histogram sum:", np.sum(histogram))
        # print("Nonzero bins:", np.count_nonzero(histogram))

        # Retrieve the top 3 best matching images
        top_matches = retrieve_similar_image_tf_idf(histogram, db_data, num_clusters, top_k=30)
        best_match, inliners, query_keypoints, best_match_keypoints, inliner_ratio = find_best_matched_image(keypoints_xy, descriptors, top_matches, db_data)
        if best_match is None:
            print("can't recognize where you are!")
        # rvec, tvec = compute_camera_pose_from_top_match(keypoints_xy, descriptors, best_match, db_data, camera_matrix)
        best_result = bf_best_pose_from_top_matches(
            keypoints_xy,
            descriptors,
            list(range(0, 34)),
            db_data,
            camera_matrix,
            top_k = 5
        )
        rvec, tvec, best_match_index, num_inliers, inliner_ratio = best_result

        R, _ = cv2.Rodrigues(rvec)
        camera_position_world = -np.dot(R.T, tvec)
        x = camera_position_world[0].item()
        y = camera_position_world[1].item()
        print(f"Estimated camera location (world coords): x={x:.2f}, y={y:.2f}")

        print(f"Best match index: {best_match_index} with {num_inliers} inliners and {inliner_ratio}% ratio")
        print(f"Estimated camera location (world coords): x={x:.2f}, y={y:.2f}")

        # --- build reply payload -------------------------------------------------
        reply_dict = {
            "x": float(x),
            "y": float(y),
            # "x":35,
            # "y":35,
            "status": "ok"
        }

        reply_json = json.dumps(reply_dict).encode("utf-8")
        reply_compressed = gzip.compress(reply_json)
        reply_len = struct.pack("!I", len(reply_compressed))

        # --- send it back to the same client socket ------------------------------
        client_socket.sendall(reply_len + reply_compressed)
        print(f"Sent {len(reply_compressed)}-byte reply to client")

    except Exception as e:
        print(f"Error: {e}")
        # --- build reply payload -------------------------------------------------
        reply_dict = {
            # "x": float(camera_position_world[0]),
            # "y": float(camera_position_world[1]),
            "x": -1,
            "y": -1,
            "status": "error"
        }

        reply_json = json.dumps(reply_dict).encode("utf-8")
        reply_compressed = gzip.compress(reply_json)
        reply_len = struct.pack("!I", len(reply_compressed))

        # --- send it back to the same client socket ------------------------------
        client_socket.sendall(reply_len + reply_compressed)
        print(f"Sent {len(reply_compressed)}-byte reply to client")
    finally:
        client_socket.close()