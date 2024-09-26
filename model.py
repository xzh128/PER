from sklearn.neighbors import BallTree
import networkx as nx
import os
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from GFM import GFM
import open3d as o3d
import itertools
import csv
from scipy.sparse.csgraph import minimum_spanning_tree
from transformer import compute_inverse_transformations
from transformer import merge_transformations
from transformer import apply_transformation_to_point_cloud, convert_matrices_to_euler_angles
from sklearn.neighbors import NearestNeighbors
class CustomGenerator(nn.Module):
    def __init__(self, embedding_dimensions):
        super(CustomGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimensions, embedding_dimensions // 2),
            nn.LayerNorm(embedding_dimensions // 2),
            nn.ELU(),
            nn.Linear(embedding_dimensions // 2, embedding_dimensions // 4),
            nn.LayerNorm(embedding_dimensions // 4),
            nn.ELU()
        )
        self.rotation_projection = nn.Linear(embedding_dimensions // 4, 4)
        self.translation_projection = nn.Linear(embedding_dimensions // 4, 3)

    def forward(self, input_tensor):
        max_input = input_tensor.max(dim=1)[0]
        transformed = self.network(max_input)
        rotation_vector = self.rotation_projection(transformed)
        translation_vector = self.translation_projection(transformed)
        normalized_rotation = rotation_vector / torch.norm(rotation_vector, p=2, dim=1, keepdim=True)
        return normalized_rotation, translation_vector

class Encoder(nn.Module):
    def __init__(self, layer, num_layers, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(layer.size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, src_mask):
        for layer in self.layer_stack:
            x = layer(x, src_mask)
            x = self.dropout(x)
        return self.layer_norm(x)
class Decoder(nn.Module):
    def __init__(self, layer, num_layers, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(layer.size)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layer_stack:
            x = layer(x, memory, src_mask, tgt_mask)
            x = self.dropout(x)
        return self.layer_norm(x)
    def decode_with_attention(self, x, memory, src_mask, tgt_mask):
        x = self.forward(x, memory, src_mask, tgt_mask)
        return F.log_softmax(x, dim=-1)
class LayerNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

        self.epsilon = epsilon
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.epsilon)
        output = self.scale * normalized + self.shift
        return output
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, attention, feed_forward, dropout_rate):
        super(EncoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.layers = self._initialize_layers(embed_size, dropout_rate)
        self.embed_size = embed_size
    def _initialize_layers(self, embed_size, dropout_rate):
        return nn.ModuleList([SublayerConnection(embed_size, dropout_rate) for _ in range(2)])

    def forward(self, input_tensor, attention_mask):
        attended = self.layers[0](input_tensor, lambda x: self.attention(x, x, x, attention_mask))
        return self.layers[1](attended, self.feed_forward)
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, self_attention, cross_attention, feed_forward, dropout_rate):
        super(DecoderBlock, self).__init__()
        self.embed_size = embed_size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.layers = self._initialize_layers(dropout_rate)
    def _initialize_layers(self, dropout_rate):
        return nn.ModuleList([SublayerConnection(self.embed_size, dropout_rate) for _ in range(3)])

    def forward(self, input_tensor, memory, source_mask, target_mask):
        attended_self = self.layers[0](input_tensor, lambda x: self.self_attention(x, x, x, target_mask))
        attended_cross = self.layers[1](attended_self, lambda x: self.cross_attention(x, memory, memory, source_mask))
        return self.layers[2](attended_cross, self.feed_forward)

class Attention(nn.Module):
    def __init__(self, num_heads, model_dim, dropout_rate=0.0):
        super(Attention, self).__init__()
        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_layers = self._initialize_linear_layers(model_dim)
        self.attention_weights = None
        self.dropout_layer = nn.Dropout(p=dropout_rate)
    def _initialize_linear_layers(self, model_dim):
        return nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(4)])

    def forward(self, query, key, value, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        scaled_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scaled_scores = scaled_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = F.softmax(scaled_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = F.threshold(output, threshold=0.2, value=0.0, inplace=True)
        output = self.linear_layers[-1](output)
        output_mean = torch.mean(output, dim=-1, keepdim=True)
        output += output_mean
        return output

class FeedForwardNN(nn.Module):
    def __init__(self, model_dim, feedforward_dim, dropout_rate=0.1):
        super(FeedForwardNN, self).__init__()
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_tensor):
        activated = F.leaky_relu(self.linear1(input_tensor), negative_slope=0.2)
        dropped = self.dropout_layer(activated)
        output = self.linear2(dropped)
        return output
class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.embedding_dim = config.n_emb_dims
        self.num_blocks = config.n_blocks
        self.dropout_rate = config.dropout
        self.feedforward_dim = config.n_ff_dims
        self.num_heads = config.n_heads
        attn_layer = Attention(self.num_heads, self.embedding_dim)
        feedforward_layer = FeedForwardNN(self.embedding_dim, self.feedforward_dim, self.dropout_rate)
        self.encoder = Encoder(
            EncoderBlock(self.embedding_dim, copy.deepcopy(attn_layer), copy.deepcopy(feedforward_layer), self.dropout_rate),
            self.num_blocks
        )
        self.decoder = Decoder(
            DecoderBlock(self.embedding_dim, copy.deepcopy(attn_layer), copy.deepcopy(attn_layer), copy.deepcopy(feedforward_layer), self.dropout_rate),
            self.num_blocks
        )

    def forward(self, source, target):
        source = source.transpose(2, 1).contiguous()
        target = target.transpose(2, 1).contiguous()
        encoded_src = self.encoder(source, None)
        decoded_tgt = self.decoder(target, encoded_src, None, None)
        return encoded_src.transpose(2, 1).contiguous(), decoded_tgt.transpose(2, 1).contiguous()

class PositionalEncoding(nn.Module):
    def __init__(self, length, scale):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Sequential(
            nn.Linear(3, length * scale),
            nn.Sigmoid(),
            nn.Linear(length * scale, length),
            nn.LeakyReLU()
        )

    def forward(self, inputs):
        return self.encoding(inputs)

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionLayer, self).__init__()
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.maximum_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        avg_output = self.average_pool(inputs)
        max_output = self.maximum_pool(inputs)
        avg_output = self.fc_layers(avg_output)
        max_output = self.fc_layers(max_output)
        return inputs * avg_output + inputs * max_output

class BiAttentionModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16, head_count=4):
        super(BiAttentionModule, self).__init__()
        self.channel_attention_layer = ChannelAttentionLayer(channels, reduction_ratio)
        self.spatial_attention_layer = nn.MultiheadAttention(embed_dim=channels, num_heads=head_count)

    def forward(self, inputs):
        channel_attended = self.channel_attention_layer(inputs)
        reshaped_inputs = inputs.permute(2, 0, 1)
        spatial_attended, _ = self.spatial_attention_layer(reshaped_inputs, reshaped_inputs, reshaped_inputs)
        spatial_attended = spatial_attended.permute(1, 2, 0)
        output = channel_attended * spatial_attended
        return output
def Knn(x, k, sigma=1.0):
    x = x.transpose(1, 2)
    pairwise_distances = torch.cdist(x, x)
    weights = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
    idx = weights.topk(k=k, dim=-1)[1]
    return idx
def TriS(pcd, pairs):
    B, N, N_p, _ = pairs.shape
    result = torch.zeros((B * N, N_p, 3), dtype=torch.float32)
    temp = (torch.arange(B) * N).reshape(B, 1, 1, 1).repeat(1, N, N_p, 3).cuda()
    pairs = pairs + temp
    pcd = pcd.reshape(-1, 3)
    pairs = pairs.reshape(-1, N_p, 3)
    edge1 = pcd[pairs[:, :, 1]] - pcd[pairs[:, :, 0]]
    edge2 = pcd[pairs[:, :, 2]] - pcd[pairs[:, :, 0]]
    normals = torch.cross(edge1, edge2, dim=-1)
    areas = 0.5 * torch.norm(normals, dim=-1)
    result[:, :, 0] = areas
    result = result.reshape(B, N, N_p, 3)
    result, _ = torch.sort(result, dim=-1, descending=False)
    return result
def GFM(src, tgt, Sharp=20, Threshold=0.5, k=10, eps=1e-6):
    batch_size, num_points = src.shape[0], src.shape[2]
    index = torch.arange(num_points).reshape(1, -1, 1).repeat(batch_size, 1, k * (k - 1) // 2).unsqueeze(-1).cuda()
    knn_indices = Knn(src, k)
    combinations = list(itertools.combinations(range(k), 2))
    knn_pairs = knn_indices[:, :, combinations]
    triangles = torch.cat((index, knn_pairs), dim=-1)
    src = src.transpose(-1, -2)
    src_lengths = TriS(src, triangles)
    tgt_lengths = TriS(tgt, triangles) + eps
    squared_diff = (src_lengths - tgt_lengths) ** 2
    loss = squared_diff.sum(dim=-1) / (src_lengths + tgt_lengths).sum(dim=-1)
    sorted_loss, _ = torch.sort(loss, dim=-1, descending=False)
    regularization_loss = torch.norm(src_lengths - tgt_lengths, dim=-1)
    total_loss = sorted_loss + 0.1 * regularization_loss
    median_loss = total_loss.median(dim=-1, keepdim=True)[0]
    mask_outliers = total_loss > 3 * median_loss
    total_loss[mask_outliers] = 0
    limited_loss = total_loss[:, :, :k]
    limited_loss = torch.sqrt(limited_loss + eps)
    mean_loss = limited_loss.mean(dim=-1)
    min_loss, _ = mean_loss.min(dim=-1, keepdim=True)
    normalized_loss = mean_loss - min_loss
    weights = 2 * torch.sigmoid(-Sharp * normalized_loss.cuda())
    weights[weights <= Threshold] = 0
    weights[weights > Threshold] = 1
    return weights
def find_matching_indices(source, target, voxel_size, K=None):
    kd_tree = o3d.geometry.KDTreeFlann(target)
    matches = []
    for i, point in enumerate(source.points):
        [_, indices, _] = kd_tree.search_radius_vector_3d(point, voxel_size)
        if K is not None:
            indices = indices[:K]
        for index in indices:
            matches.append((i, index))
    return matches
def duplicate_module(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])
def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity_matrix = torch.eye(3, device=rotation_ab.device).unsqueeze(0).expand(batch_size, -1, -1)
    rotation_loss = F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity_matrix)
    translation_loss = F.mse_loss(translation_ab, -translation_ba)
    return rotation_loss + translation_loss + F.mse_loss(translation_ab, -translation_ba)
class TempNet(nn.Module):
    def __init__(self, config):
        super(TempNet, self).__init__()
        self.embedding_dim = config.n_emb_dims
        self.temp_scale = config.temp_factor
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.ELU()
        )
        self.feature_difference = None
    def forward(self, *inputs):
        src_emb = inputs[0].mean(dim=2)
        tgt_emb = inputs[1].mean(dim=2)
        diff = torch.abs(src_emb - tgt_emb)
        self.feature_difference = diff
        temperature = torch.clamp(self.network(diff), 1.0 / self.temp_scale, 1.0 * self.temp_scale)
        return temperature, diff
def procrustes_analysis(X, Y, weights, epsilon):
    assert len(X) == len(Y), "Input shapes must match"
    total_weight = torch.abs(weights).sum()
    weights = weights.reshape(-1, 1).repeat(1, 3)
    normalized_weights = weights / (total_weight + epsilon)
    mean_x = (normalized_weights * X).sum(dim=0, keepdim=True)
    mean_y = (normalized_weights * Y).sum(dim=0, keepdim=True)
    covariance_matrix = (Y - mean_y).t() @ (normalized_weights * (X - mean_x)).cpu().double()
    U, _, V = covariance_matrix.svd()
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1
    rotation = U @ S @ V.t()
    translation = (mean_y.squeeze() - rotation @ mean_x.t()).float()
    return rotation.float(), translation

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.corres_mode = args.corres_mode
        self.Sharp = args.GMCCE_Sharp
        self.Threshold = args.GMCCE_Thres
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.my_iter = torch.ones(1)
        self.last_tem = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size()
        temperature = input[4].view(batch_size, 1, 1)
        is_corr = input[5]

        if self.corres_mode == True and is_corr == True:
            R = []
            T = []
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(10000 * scores, dim=2)
            _, corres = torch.max(scores, dim=-1)
            corr_tgt = torch.matmul(scores, tgt.transpose(1, 2))
            corres = corres.reshape(corres.shape[0], -1, 1)
            weight = GFM(src, corr_tgt, Sharp=self.Sharp, Threshold=self.Threshold)
            weight = weight.unsqueeze(-1)
            src = src.transpose(1, 2).contiguous()
            tgt = tgt.transpose(1, 2).contiguous()
            for i in range(src.shape[0]):
                src_corr = tgt[i][corres[i]].squeeze()
                r, t = procrustes_analysis(src[i].cpu(), src_corr.cpu(), weight[i].detach().cpu(), 1e-7)
                R.append(r)
                T.append(t)
            R = torch.stack(R, dim=0).cuda()
            T = torch.stack(T, dim=0).cuda()
            return R, T, corres, weight
        else:
            R = []
            T = []
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(temperature * scores, dim=2)
            idx = torch.arange(src.shape[2]).reshape(1, -1).repeat(src.shape[0], 1)
            weight, corres = torch.max(scores, dim=-1)
            corres = corres.reshape(corres.shape[0], -1, 1)
            weight = weight.unsqueeze(-1)
            src = src.transpose(1, 2).contiguous()
            tgt = tgt.transpose(1, 2).contiguous()
            for i in range(src.shape[0]):
                src_corr = tgt[i][corres[i]].squeeze()
                r, t = procrustes_analysis(src[i].cpu(), src_corr.cpu(), weight[i].detach().cpu(), 1e-7)
                R.append(r)
                T.append(t)
            R = torch.stack(R, dim=0).cuda()
            T = torch.stack(T, dim=0).cuda()
            return R, T, corres, weight

class Prediction(nn.Module):
    def __init__(self, args):
        super(Prediction, self).__init__()
        self.embedding_dim = args.n_emb_dims
        self.position_encoding = PositionalEncoding(args.n_emb_dims, 8)
        self.se_block = BiAttentionModule(ch_in=args.n_emb_dims)
        self.embedding_network = GFM(embed_dim=args.n_emb_dims, token_dim=args.token_dim)
        self.transformer_model = TransformerModel(args=args)
        self.temperature_net = TempNet(args)
        self.svd_head = SVDHead(args=args)

    def forward(self, *inputs):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr = self.predict_embedding(*inputs)
        rotation_ab, translation_ab, corres_ab, weight_ab = self.svd_head(src_embedding, tgt_embedding, src, tgt, temperature, is_corr)
        rotation_ba, translation_ba, corres_ba, weight_ba = self.svd_head(tgt_embedding, src_embedding, tgt, src, temperature, is_corr)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab

    def predict_embedding(self, *inputs):
        src = inputs[0]
        tgt = inputs[1]
        is_corr = inputs[2]
        src_embedding = self.embedding_network(src)
        tgt_embedding = self.embedding_network(tgt)
        src_encoding = self.position_encoding(src.transpose(1, 2)).transpose(1, 2).contiguous()
        tgt_encoding = self.position_encoding(tgt.transpose(1, 2)).transpose(1, 2).contiguous()
        src_embedding_p, tgt_embedding_p = self.transformer_model(src_embedding + src_encoding, tgt_embedding + tgt_encoding)
        src_embedding = self.se_block(src_embedding + src_embedding_p)
        tgt_embedding = self.se_block(tgt_embedding + tgt_embedding_p)
        temperature, feature_disparity = self.temperature_net(src_embedding, tgt_embedding)
        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr

    def predict_keypoint_correspondence(self, *inputs):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*inputs)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size * num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
            pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def validate_correspondences(matches, correspondences):
    batch_size, num_points, _ = correspondences.shape
    index_src = torch.arange(num_points).reshape(1, -1, 1).repeat(batch_size, 1, 1)
    extended_correspondences = torch.cat([index_src, correspondences], dim=-1).int()
    correctness_flags = []

    for match, corres_iter in zip(matches, extended_correspondences):
        is_correct_iter = verify_correspondences(
            torch.tensor(match).int().unsqueeze(0),
            corres_iter.unsqueeze(0),
            num_points
        )
        correctness_flags.extend(is_correct_iter)
    return np.array(correctness_flags)

def identify_correspondences(source, target, rotation_matrix, translation_vector):
    transformed_source = apply_transformation_to_point_cloud(source, rotation_matrix, translation_vector)
    correspondences = []

    for point_cloud_src, point_cloud_tgt in zip(transformed_source, target):
        pcd_src = make_open3d_point_cloud(point_cloud_src.cpu().numpy().T)
        pcd_tgt = make_open3d_point_cloud(point_cloud_tgt.cpu().numpy().T)
        matches = find_matching_indices(pcd_src, pcd_tgt, 0.04)
        correspondences.append(matches)
    return correspondences
def compute_hash(array, modulus=None):
    if isinstance(array, np.ndarray):
        num_points, dimensions = array.shape
    else:
        num_points, dimensions = len(array[0]), len(array)
    hash_vector = np.zeros(num_points, dtype=np.int64)
    for dim in range(dimensions):
        if isinstance(array, np.ndarray):
            hash_vector += array[:, dim] * modulus ** dim
        else:
            hash_vector += array[dim] * modulus ** dim
    return hash_vector
def verify_correspondences(positive_pairs, predicted_pairs, hash_seed=None):
    assert len(positive_pairs) == len(predicted_pairs)
    correctness = []
    for pos_pair, pred_pair in zip(positive_pairs, predicted_pairs):
        if isinstance(pos_pair, torch.Tensor):
            pos_pair = pos_pair.numpy()
        if isinstance(pred_pair, torch.Tensor):
            pred_pair = pred_pair.numpy()
        pos_keys = compute_hash(pos_pair, hash_seed)
        pred_keys = compute_hash(pred_pair, hash_seed)
        correctness.append(np.isin(pred_keys, pos_keys, assume_unique=False))
    return np.hstack(correctness)

# The following code comes from a reference
class GPR(nn.Module):
    def __init__(self, args):
        super(GPR, self).__init__()
        self.num_iterations = args.n_iters
        self.logger = Logger(args)
        self.discount_factor = args.discount_factor
        self.discrimination_loss = args.discrimination_loss
        self.temp_prediction_model = Prediction(args)
        self.model_path = args.model_path
        self.cycle_consistency_loss = args.cycle_consistency_loss
        self.criterion = nn.BCELoss()
        if torch.cuda.device_count() > 1:
            self.temp_prediction_model = nn.DataParallel(self.temp_prediction_model)
    def forward(self, *input):
        rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab = self.temp_prediction_model(
            *input)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab

    def predict(self, source, target, n_iters=2):
        batch_size = source.size(0)
        rotation_ab_pred = torch.eye(3, device=source.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(batch_size, 3, device=source.device, dtype=torch.float32)
        for _ in range(n_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, _ = self.forward(
                source, target)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred += torch.matmul(rotation_ab_pred_i, translation_ab_pred_i.unsqueeze(2)).squeeze(2)
            source = apply_transformation_to_point_cloud(source, rotation_ab_pred_i, translation_ab_pred_i)
        return rotation_ab_pred, translation_ab_pred

    def train_one_batch(self, pcd, Transform, opt):
        opt.zero_grad()
        src, tgt = pcd['src'], pcd['tgt']
        rotation_ab, translation_ab = Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation_ab_pred = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(batch_size, 3, device=device)
        rotation_ba_pred = rotation_ab_pred.clone()
        translation_ba_pred = translation_ab_pred.clone()
        total_loss = 0
        total_cycle_consistency_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}
        corres_gt = identify_correspondences(src, tgt, rotation_ab, translation_ab)
        for i in range(self.num_iters):
            is_corr = 1
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
                feature_disparity, corres_ab, weight_ab = self.forward(src, tgt, is_corr)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = translation_ab_pred + torch.matmul(rotation_ab_pred_i,
                                                                     translation_ab_pred_i.unsqueeze(2)).squeeze(2)

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = translation_ba_pred + torch.matmul(rotation_ba_pred_i,
                                                                     translation_ba_pred_i.unsqueeze(2)).squeeze(2)

            src = apply_transformation_to_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            mse_loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) +
                        F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += mse_loss
            is_correct = torch.tensor(validate_correspondences(corres_gt, corres_ab.cpu())).squeeze()
            accuracy = is_correct.sum().float() / is_correct.shape[0]
            if self.discrimination_loss != 0:
                discrimination_loss = self.discrimination_loss * self.crit(weight_ab.view(-1, 1).cpu(),
                                                                           is_correct.float()) * self.discount_factor ** i
                total_discrimination_loss += discrimination_loss
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i,
                                                       translation_ba_pred_i) * self.cycle_consistency_loss * self.discount_factor ** i
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss += mse_loss + cycle_consistency_loss + total_discrimination_loss
        total_loss.backward()
        opt.step()
        losses.update({
            'total_loss': total_loss.item(),
            'cycle': total_cycle_consistency_loss.item(),
            'scale': 0,
            'acc': accuracy.item(),
            'discrimination': total_discrimination_loss.item(),
            'mse': total_mse_loss.item()
        })

        Transforms_Pred['R_ab_pred'] = rotation_ab_pred
        Transforms_Pred['T_ab_pred'] = translation_ab_pred
        return losses, Transforms_Pred

    def _test_one_batch(self, pcd, Transform, vis):
        src, tgt = pcd['src'], pcd['tgt']
        rotation_ab, translation_ab = Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(batch_size, 3, device=device)
        rotation_ba_pred = rotation_ab_pred.clone()
        translation_ba_pred = translation_ab_pred.clone()

        total_loss = 0
        total_cycle_consistency_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}

        corres_gt = identify_correspondences(src, tgt, rotation_ab, translation_ab)

        if vis:
            self.Visulization(pcd)

        for i in range(self.num_iters):
            is_corr = (i != 0)
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
                feature_disparity, corres_ab, weight_ab = self.forward(src, tgt, is_corr)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred += torch.matmul(rotation_ab_pred_i, translation_ab_pred_i.unsqueeze(2)).squeeze(2)
            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred += torch.matmul(rotation_ba_pred_i, translation_ba_pred_i.unsqueeze(2)).squeeze(2)
            src = apply_transformation_to_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) +
                    F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += loss

            is_correct = torch.tensor(validate_correspondences(corres_gt, corres_ab.cpu())).squeeze()
            accuracy = is_correct.sum().float() / is_correct.shape[0]

            discrimination_loss = 0
            if self.discrimination_loss != 0:
                discrimination_loss = self.discrimination_loss * self.crit(weight_ab.view(-1, 1).cpu(),
                                                                           is_correct.float()) * self.discount_factor ** i
                total_discrimination_loss += discrimination_loss

            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i,
                                                       translation_ba_pred_i) * self.cycle_consistency_loss * self.discount_factor ** i
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss += loss + cycle_consistency_loss + discrimination_loss
            if vis:
                pcd['src'] = src
                self.Visulization(pcd)
            losses['total_loss'] = total_loss.item()
            losses['cycle'] = total_cycle_consistency_loss.item()
            losses['scale'] = 0
            losses['acc'] = accuracy.item()
            losses['discrimination'] = total_discrimination_loss.item()
            losses['mse'] = total_mse_loss.item()

            Transforms_Pred['R_ab_pred'] = rotation_ab_pred
            Transforms_Pred['T_ab_pred'] = translation_ab_pred

        return losses, Transforms_Pred

    def Visulization(self, pcd):
        src = pcd['src']
        tgt = pcd['tgt']
        pcd0 = make_open3d_point_cloud(src[0].cpu().numpy().T, color=[0, 255, 0])
        pcd1 = make_open3d_point_cloud(tgt[0].cpu().numpy().T, color=[255, 0, 0])
        o3d.visualization.draw_geometries([pcd0, pcd1])

    def Compute_metrics(self, avg_losses, Transforms, mode):
        concatenated = merge_transformations(compute_inverse_transformations(Transforms['R_ab'], Transforms['T_ab']),
                                   np.concatenate(
                                               [Transforms['R_ab_pred'], Transforms['T_ab_pred'].unsqueeze(-1)],
                                               axis=-1))
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = np.mean(
            (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).numpy())
        residual_transmag = np.mean((concatenated[:, :, 3].norm(dim=-1)).numpy())

        rotations_ab = Transforms['R_ab'].numpy()
        translations_ab = Transforms['T_ab'].numpy()
        rotations_ab_pred = Transforms['R_ab_pred'].numpy()
        translations_ab_pred = Transforms['T_ab_pred'].numpy()

        r_ab_mae = np.abs(Transforms['euler_ab'] - Transforms['euler_ab_pred'])
        t_ab_mae = np.abs(translations_ab - translations_ab_pred)
        if mode['save_mae']:
            error_result = []
            error_result.append(r_ab_mae)
            error_result.append(t_ab_mae)
            error_result = np.array(error_result)
            np.save(mode['outfile'], error_result)
        if mode['cur_list']:
            cur_accs = []
            threshold = np.arange(0, 10, 0.01)
            with open('./test.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(threshold.shape[0]):
                    cur_acc = np.mean((r_ab_mae <= threshold[i] / 100000) * (t_ab_mae <= 0.1))
                    cur_accs.append(cur_acc)
                writer.writerow(cur_accs)
                cur_accs = []
                for i in range(threshold.shape[0]):
                    cur_acc = np.mean((r_ab_mae <= 5) * (t_ab_mae <= threshold[i] / 10000000))
                    cur_accs.append(cur_acc)
                writer.writerow(cur_accs)
        cur_acc = np.mean((r_ab_mae <= 1) * (t_ab_mae <= 0.01))
        r_ab_mse = np.mean((Transforms['euler_ab'] - Transforms['euler_ab_pred']) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(r_ab_mae)
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(t_ab_mae)
        r_ab_r2_score = r2_score(Transforms['euler_ab'], Transforms['euler_ab_pred'])
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

        info = {'arrow': 'A->B',
                'epoch': avg_losses['epoch'],
                'stage': avg_losses['stage'],
                'loss': avg_losses['avg_loss'],
                'cycle_consistency_loss': avg_losses['avg_cycle'],
                'scale_consensus_loss': avg_losses['avg_scale'],
                'dis_loss': avg_losses['avg_discrimination'],
                'mse_loss': avg_losses['avg_mse'],
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score,
                'corres_accuracy': avg_losses['avg_acc'],
                'r_ab_mie': residual_rotdeg,
                't_ab_mie': residual_transmag,
                'cur_acc': cur_acc}
        self.logger.write(info)
        return info
    def _train_one_epoch(self, epoch, train_loader, opt, args):
        self.train()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        total_corres_accuracy = 0.0
        total_cycle_consistency_loss = 0.0
        total_scale_consensus_loss = 0.0
        total_mse_loss = 0.0
        total_discrimination_loss = 0.0
        avg_losses = {}
        Transforms = {}
        Metrics_mode = {}
        vis = False
        for pcd, Transform in tqdm(train_loader):
            for key in pcd.keys():
                pcd[key] = pcd[key].cuda()
            for key in Transform.keys():
                Transform[key] = Transform[key].cuda()
            losses, Transform_pred = self._train_one_batch(pcd, Transform, opt)
            batch_size = pcd['src'].size(0)
            num_examples += batch_size
            total_mse_loss += losses['mse'] * batch_size
            total_loss = total_loss + losses['total_loss'] * batch_size
            total_cycle_consistency_loss = total_cycle_consistency_loss + losses['cycle'] * batch_size
            total_scale_consensus_loss = total_scale_consensus_loss + losses['scale'] * batch_size
            total_corres_accuracy += losses['acc'] * batch_size
            total_discrimination_loss += losses['discrimination'] * batch_size
            rotations_ab.append(Transform['R_ab'].detach().cpu())
            translations_ab.append(Transform['T_ab'].detach().cpu())
            rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
            translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())
            eulers_ab.append(Transform['euler_ab'].cpu().numpy())

        avg_losses['avg_loss'] = total_loss / num_examples
        avg_losses['avg_cycle'] = total_cycle_consistency_loss / num_examples
        avg_losses['avg_scale'] = total_scale_consensus_loss / num_examples
        avg_losses['avg_acc'] = total_corres_accuracy / num_examples
        avg_losses['avg_discrimination'] = total_discrimination_loss / num_examples
        avg_losses['avg_mse'] = total_mse_loss / num_examples
        avg_losses['epoch'] = epoch
        avg_losses['stage'] = 'train'
        Transforms['R_ab'] = torch.cat(rotations_ab, axis=0)
        Transforms['T_ab'] = torch.cat(translations_ab, axis=0)
        Transforms['R_ab_pred'] = torch.cat(rotations_ab_pred, axis=0)
        Transforms['T_ab_pred'] = torch.cat(translations_ab_pred, axis=0)
        Transforms['euler_ab'] = np.degrees(np.concatenate(eulers_ab, axis=0))
        Transforms['euler_ab_pred'] = convert_matrices_to_euler_angles(Transforms['R_ab_pred'])
        Metrics_mode['cur_list'] = True
        Metrics_mode['save_mae'] = True
        Metrics_mode['outfile'] = 'mae.npy'
        return self.Compute_metrics(avg_losses, Transforms, Metrics_mode)

    def _test_one_epoch(self, epoch, test_loader, vis):
        with torch.no_grad():
            self.eval()
            total_loss = 0
            rotations_ab = []
            translations_ab = []
            rotations_ab_pred = []
            translations_ab_pred = []
            eulers_ab = []
            num_examples = 0
            total_corres_accuracy = 0.0
            total_cycle_consistency_loss = 0.0
            total_scale_consensus_loss = 0.0
            total_mse_loss = 0.0
            total_discrimination_loss = 0.0
            avg_losses = {}
            Transforms = {}
            Metrics_mode = {}
            for pcd, Transform in tqdm(test_loader):
                for key in pcd.keys():
                    pcd[key] = pcd[key].cuda()
                for key in Transform.keys():
                    Transform[key] = Transform[key].cuda()
                losses, Transform_pred = self._test_one_batch(pcd, Transform, vis)
                batch_size = pcd['src'].size(0)
                num_examples += batch_size
                total_mse_loss += losses['mse'] * batch_size
                total_loss = total_loss + losses['total_loss'] * batch_size
                total_cycle_consistency_loss = total_cycle_consistency_loss + losses['cycle'] * batch_size
                total_scale_consensus_loss = total_scale_consensus_loss + losses['scale'] * batch_size
                total_corres_accuracy += losses['acc'] * batch_size
                total_discrimination_loss += losses['discrimination'] * batch_size
                rotations_ab.append(Transform['R_ab'].detach().cpu())
                translations_ab.append(Transform['T_ab'].detach().cpu())
                rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
                translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())
                eulers_ab.append(Transform['euler_ab'].cpu().numpy())
            avg_losses['avg_loss'] = total_loss / num_examples
            avg_losses['avg_cycle'] = total_cycle_consistency_loss / num_examples
            avg_losses['avg_scale'] = total_scale_consensus_loss / num_examples
            avg_losses['avg_acc'] = total_corres_accuracy / num_examples
            avg_losses['avg_discrimination'] = total_discrimination_loss / num_examples
            avg_losses['avg_mse'] = total_mse_loss / num_examples
            avg_losses['epoch'] = epoch
            avg_losses['stage'] = 'test'
            Transforms['R_ab'] = torch.cat(rotations_ab, axis=0)
            Transforms['T_ab'] = torch.cat(translations_ab, axis=0)
            Transforms['R_ab_pred'] = torch.cat(rotations_ab_pred, axis=0)
            Transforms['T_ab_pred'] = torch.cat(translations_ab_pred, axis=0)
            Transforms['euler_ab'] = np.degrees(np.concatenate(eulers_ab, axis=0))
            Transforms['euler_ab_pred'] = convert_matrices_to_euler_angles(Transforms['R_ab_pred'])
            Metrics_mode['cur_list'] = False
            Metrics_mode['save_mae'] = False
            Metrics_mode['outfile'] = 'Ours_clean_mae.npy'
        return self.Compute_metrics(avg_losses, Transforms, Metrics_mode)
    def save(self, path):
        if torch.cuda.device_count() > 1:
            torch.save(self.T_prediction.module.state_dict(), path)
        else:
            torch.save(self.T_prediction.state_dict(), path)

    def load(self, path):
        self.T_prediction.load_state_dict(torch.load(path))

class Logger:
    def __init__(self, args):
        self.path = os.path.join('checkpoints', args.exp_name)
        os.makedirs(self.path, exist_ok=True)
        self.log_file = os.path.join(self.path, 'log')
        with open(self.log_file, 'a') as fw:
            fw.write(str(args) + '\n')
        print(str(args))
        with open(os.path.join(self.path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    def write(self, info):
        text = (
            f"{info['arrow']}:: Stage: {info['stage']}, Epoch: {info['epoch']}, Loss: {info['loss']:.6f}, "
            f"Cycle_consistency_loss: {info['cycle_consistency_loss']:.6f}, "
            f"Scale_consensus_loss: {info['scale_consensus_loss']:.6f}, dis_loss: {info['dis_loss']:.6f}, "
            f"mse_loss: {info['mse_loss']:.6f}, Rot_MSE: {info['r_ab_mse']:.10f}, "
            f"Rot_RMSE: {info['r_ab_rmse']:.10f}, Rot_MAE: {info['r_ab_mae']:.10f}, "
            f"Rot_R2: {info['r_ab_r2_score']:.6f}, Trans_MSE: {info['t_ab_mse']:.10f}, "
            f"Trans_RMSE: {info['t_ab_rmse']:.10f}, Trans_MAE: {info['t_ab_mae']:.10f}, "
            f"Trans_R2: {info['t_ab_r2_score']:.6f}, corres_accuracy: {info['corres_accuracy']:.6f}, "
            f"r_ab_mie: {info['r_ab_mie']:.10f}, t_ab_mie: {info['t_ab_mie']:.10f}, cur_acc: {info['cur_acc']:.6f}\n"
        )
        with open(self.log_file, 'a') as fw:
            fw.write(text)
        print(text)
    def close(self):
        pass
if __name__ == '__main__':
    print('OK')
