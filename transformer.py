from timm.models.layers import DropPath
import numpy as np
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import torch
import torch.nn as nn
class AffineTransform(nn.Module):
    def __init__(self, dim):
        super(AffineTransform, self).__init__()
        self.transform = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.transform(x) + self.bias

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.layers(x) + x

class ResidualSqueezeMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation=nn.GELU, dropout_rate=0.):
        super().__init__()
        output_dim = input_dim+hidden_dim
        hidden_dim = input_dim+output_dim

        self.layers = nn.Sequential(
            AffineTransform(input_dim),
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_rate)
        )
        self.residual1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual2 = ResidualBlock(output_dim, output_dim)

    def forward(self, x):
        x_transformed = self.layers(x)
        x_residual = self.residual1(x_transformed)
        x_residual = self.residual2(x_residual)
        return x + x_residual

class AttentionWithSoftThreshold(nn.Module):
    def __init__(self, dim, num_heads=8, input_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 norm=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = norm(input_dim, eps=0.0001)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.input_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.input_dim)
        v = v.transpose(1, 2).reshape(B, N, self.input_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = v.squeeze(1) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, input_dim, num_heads, mlp_ratio=1.0, qkv_bias=False, qk_scale=None, dropout=0.0, attention_dropout=0.0,
                 drop_path=0.0, activation=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.layer_norm1 = norm(dim)
        self.attention = AttentionWithSoftThreshold(
            dim, input_dim=input_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attention_drop=attention_dropout, projection_drop=dropout)
        self.path_dropout = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_norm2 = norm(input_dim)
        self.mlp = ResidualSqueezeMlp(
            input_features=input_dim, hidden_features=int(input_dim * mlp_ratio), output_features=input_dim,
            act_layer=activation, drop=dropout)

    def forward(self, x):
        normalized_input = self.layer_norm1(x)
        attended_output = self.attention(normalized_input)
        normalized_attention = self.layer_norm2(attended_output)
        return normalized_attention + self.path_dropout(self.mlp(normalized_attention))

def compute_inverse_transformations(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    inv_rotation = np.transpose(rotation, axes=(0, 2, 1))
    inv_translation = -inv_rotation @ translation[..., None]
    inverse_transform = np.concatenate([inv_rotation, inv_translation], axis=-1)
    return inverse_transform

def merge_transformations(transform_a: np.ndarray, transform_b: np.ndarray) -> torch.Tensor:
    rotation_a, translation_a = transform_a[..., :3, :3], transform_a[..., :3, 3]
    rotation_b, translation_b = transform_b[..., :3, :3], transform_b[..., :3, 3]
    combined_rotation = rotation_a @ rotation_b
    combined_translation = rotation_a @ translation_b[..., None] + translation_a[..., None]
    merged_transform = np.concatenate([combined_rotation, combined_translation], axis=-1)
    return torch.tensor(merged_transform)

def convert_matrices_to_euler_angles(matrices: np.ndarray, sequence: str = 'zyx') -> np.ndarray:
    euler_angles = []
    for matrix in matrices:
        rotation = Rotation.from_matrix(matrix)
        euler_angles.append(rotation.as_euler(sequence, degrees=True))
    return np.array(euler_angles, dtype='float32')

def apply_transformation_to_point_cloud(point_cloud: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    rotated_cloud = torch.matmul(rotation, point_cloud)
    transformed_cloud = rotated_cloud + translation.unsqueeze(2)
    return transformed_cloud


