import torch.nn as nn
from transformer import Transformer
import torch.nn.functional as F
from scipy.fftpack import dct
import torch

def GFMKnn(x, k, sigma=1.0):
    x = x.transpose(1, 2)
    pairwise_distances = torch.cdist(x, x)
    weights = torch.exp(-pairwise_distances.pow(2) / (2 * sigma ** 2))
    idx = weights.topk(k=k, dim=-1)[1]
    return idx

def extract_features_from_point_cloud(point_cloud: torch.Tensor, reference_indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, num_dims = point_cloud.size()
    flattened_points = point_cloud.view(batch_size * num_points, num_dims)
    index_base = torch.arange(0, batch_size, device=point_cloud.device).view(-1, 1, 1) * num_points
    indices = reference_indices + index_base
    indices = indices.view(-1)
    features = flattened_points[indices, :]
    return features.view(batch_size, num_points, -1)

class DCTEnhancement(nn.Module):
    def __init__(self, in_dim):
        super(DCTEnhancement, self).__init__()
        self.in_dim = in_dim
    def forward(self, x):
        batch_size, num_points, num_dims = x.size()
        x_flat = x.view(batch_size * num_points, num_dims)
        x_np = x_flat.cpu().numpy()
        x_dct_np = dct(x_np, axis=1, norm='ortho')
        x_dct = torch.tensor(x_dct_np, device=x.device)
        x_dct = x_dct.view(batch_size, num_points, num_dims)
        return x_dct

class GFM(nn.Module):
    def __init__(self, num_points=20, input_channels=3, embedding_dim=256, token_dim=64, normalization_layer=nn.LayerNorm):
        super().__init__()
        self.num_points = num_points
        self.attention_layers = nn.ModuleList([
            Transformer(dim=input_channels * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0),
            Transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0),
            Transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0),
            Transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        ])
        self.projector = nn.Linear(token_dim * 4, embedding_dim)
        self.normalization = normalization_layer(embedding_dim)
        self.dct_enhancement = DCTEnhancement(input_channels)

    def forward(self, x):
        reference_indices = GFMKnn(x, self.num_points).to(x.device)
        x = x.transpose(1, 2).contiguous()
        x_dct = self.dct_enhancement(x)
        x = self.graph_feature(x_dct, reference_indices)
        attention_outputs = []
        for attention_layer in self.attention_layers:
            x = self.graph_feature(x, reference_indices)
            attention_output = attention_layer(x)
            attention_outputs.append(attention_output)
        combined_features = F.leaky_relu(torch.cat(attention_outputs, dim=-1))
        x = self.normalization(combined_features)
        return x.transpose(-1, -2).contiguous()
    def graph_feature(self, x, reference_indices):
        return extract_features_from_point_cloud(x, reference_indices)

