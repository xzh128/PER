import os
import numpy as np
import h5py
from stl import mesh
from sklearn.neighbors import NearestNeighbors


def calculate_point_density(points, neighbors_count=5):
    neighbor_finder = NearestNeighbors(n_neighbors=neighbors_count, algorithm='auto')
    neighbor_finder.fit(points)
    distances, _ = neighbor_finder.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    density_values = 1.0 / (mean_distances + 1e-6)
    return density_values

def sample_max_density(points, desired_count):
    total_points = points.shape[0]
    selected_mask = np.zeros(total_points, dtype=bool)
    initial_index = np.random.randint(0, total_points)
    selected_mask[initial_index] = True
    distances = np.sum((points - points[initial_index]) ** 2, axis=1)
    density_values = calculate_point_density(points)
    for _ in range(desired_count - 1):
        farthest_point_index = np.argmax(distances * density_values * ~selected_mask)
        selected_mask[farthest_point_index] = True
        distances = np.minimum(distances, np.sum((points - points[farthest_point_index]) ** 2, axis=1))
    return np.where(selected_mask)[0]

def stl_to_h5(input_stl_file, output_h5_file, target_points=1000, num_samples=100, label_prob=[0.8, 0.2]):
    your_mesh = mesh.Mesh.from_file(input_stl_file)
    point_cloud_data = your_mesh.vectors.reshape(-1, 3)
    mean = np.mean(point_cloud_data, axis=0)
    std_dev = np.std(point_cloud_data, axis=0)
    normalized_point_cloud_data = (point_cloud_data - mean) / std_dev
    temp_h5_path = '/.h5'
    with h5py.File(temp_h5_path, 'w') as hf:
        hf.create_dataset('data', data=normalized_point_cloud_data)
    with h5py.File(temp_h5_path, 'r') as hf:
        point_cloud = hf['data'][:]
    indices = sample_max_density(point_cloud, target_points)
    downsampled_point_cloud = point_cloud[indices]
    new_shape = (num_samples, target_points, 3)
    new_point_cloud_data = np.zeros(new_shape)
    num_samples = min(num_samples, downsampled_point_cloud.shape[0])
    new_point_cloud_data[:num_samples, :, :] = np.tile(downsampled_point_cloud, (num_samples, 1, 1))
    labels = np.random.choice([1, 2], size=(num_samples, 1), p=label_prob)
    with h5py.File(output_h5_file, 'w') as hf:
        hf.create_dataset('data', data=new_point_cloud_data)
        hf.create_dataset('label', data=labels)
    os.remove(temp_h5_path)

input_stl_file = "/home/xxx/.stl"
output_h5_file = "result.h5"
stl_to_h5(input_stl_file, output_h5_file)
