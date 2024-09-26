import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import copy
import os

def download():
    """Create a data directory if it doesn't exist."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

def load_data(partition):
    """Load data and labels from HDF5 files corresponding to a specific partition.
    Args:
        partition (str): The partition identifier for the data files.
    Returns:
        tuple: A tuple containing the concatenated data and labels.
    """
    download()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    all_data = []
    all_labels = []
    for h5_name in glob.glob(os.path.join(data_dir, 'your_file', f'data_bone_{partition}*.h5')):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_labels.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_data, all_labels


def farthest_subsample_points(pc1, pc2, num_samples=768):
    """Use farthest point sampling to select points from two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud (N, 3).
        pc2 (np.ndarray): Second point cloud (M, 3).
        num_samples (int): Desired number of points to sample.

    Returns:
        tuple: Two subsampled point clouds.
    """
    pc1 = pc1.T
    pc2 = pc2.T

    nbrs1 = NearestNeighbors(n_neighbors=num_samples, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pc1)
    random_start1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * 2
    idx1 = nbrs1.kneighbors(random_start1, return_distance=False).reshape((num_samples,))

    nbrs2 = NearestNeighbors(n_neighbors=num_samples, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pc2)
    random_start2 = np.random.random(size=(1, 3)) + np.array([[-500, -500, -500]]) * 2
    idx2 = nbrs2.kneighbors(random_start2, return_distance=False).reshape((num_samples,))
    return pc1[idx1, :].T, pc2[idx2, :].T
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.001):
    """Add Gaussian noise to a point cloud for augmentation.

    Args:
        pointcloud (np.ndarray): Original point cloud (N, C).
        sigma (float): Standard deviation of Gaussian noise.
        clip (float): Maximum magnitude of noise clipping.

    Returns:
        np.ndarray: Jittered point cloud.
    """
    noise = np.clip(sigma * np.random.randn(*pointcloud.shape), -clip, clip)
    return pointcloud + noise


import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class MedShapeNet(Dataset):
    """Dataset for medical shape analysis, allowing for rotation and noise augmentation."""

    def __init__(self, num_points, num_subsampled_points=729, partition='train',
                 gaussian_noise=0, unseen=False, rot_factor=4, category=None):
        super(MedShapeNet, self).__init__()
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.subsampled = num_points != num_subsampled_points
        if self.unseen:
            if self.partition == 'test':
                self.data = self.data[self.label >= 2]
                self.label = self.label[self.label >= 2]
            elif self.partition == 'train':
                self.data = self.data[self.label <= 1]
                self.label = self.label[self.label <= 1]

    def __getitem__(self, item):
        """Retrieve a point cloud and its transformation."""
        pointcloud = copy.deepcopy(self.data[item][:self.num_points])
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        angley = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        anglez = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(anglex), -np.sin(anglex)],
                       [0, np.sin(anglex), np.cos(anglex)]])

        Ry = np.array([[np.cos(angley), 0, np.sin(angley)],
                       [0, 1, 0],
                       [-np.sin(angley), 0, np.cos(angley)]])

        Rz = np.array([[np.cos(anglez), -np.sin(anglez), 0],
                       [np.sin(anglez), np.cos(anglez), 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        R_ba = R_ab.T
        translation_ab = np.random.uniform(-0.5, 0.5, size=3)
        translation_ba = -R_ba @ translation_ab
        pointcloud1 = pointcloud.T
        pointcloud2 = copy.deepcopy(pointcloud1)
        if self.gaussian_noise != 0:
            pointcloud1 = jitter_pointcloud(pointcloud1, clip=self.gaussian_noise)
            pointcloud2 = jitter_pointcloud(pointcloud2, clip=self.gaussian_noise)
        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud2.T).T + translation_ab[None, :]
        euler_ab = np.array([anglez, angley, anglex], dtype='float32')
        euler_ba = -euler_ab[::-1]
        index = np.random.permutation(pointcloud2.shape[0])
        pointcloud2 = pointcloud2[index]
        pcd = {
            'src': pointcloud1.astype('float32'),
            'tgt': pointcloud2.astype('float32')
        }

        Transform = {
            'R_ab': R_ab.astype('float32'),
            'T_ab': translation_ab.astype('float32'),
            'euler_ab': euler_ab,
            'R_ba': R_ba.astype('float32'),
            'T_ba': translation_ba.astype('float32'),
            'euler_ba': euler_ba
        }

        return pcd, Transform

    def __len__(self):
        return self.data.shape[0]



