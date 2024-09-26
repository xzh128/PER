
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
def knn_tri(x, k, sigma=1.0):
    x = x.transpose(1, 2)
    pairwise_distances = torch.cdist(x, x)
    weights = torch.exp(-pairwise_distances.pow(2) / (2 * sigma ** 2))
    weights = weights
    idx = weights.topk(k=k, dim=-1)[1]
    return idx
def getTri(pcd, pairs):
    B, N, N_p, _ = pairs.shape
    result = torch.zeros((B * N, N_p, 3), dtype=torch.float32)
    temp = (torch.arange(B) * N).reshape(B, 1, 1, 1).repeat(1, N, N_p, 3).cuda()
    pairs = pairs + temp
    pcd = pcd.reshape(-1, 3)
    pairs = pairs.reshape(-1, N_p, 3)
    edge1 = pcd[pairs[:, :, 1], :] - pcd[pairs[:, :, 0], :]
    edge2 = pcd[pairs[:, :, 2], :] - pcd[pairs[:, :, 0], :]
    normals = torch.cross(edge1, edge2, dim=-1)
    areas = 0.5 * torch.norm(normals, dim=-1)
    result[:, :, 0] = areas
    result = result.reshape(B, N, N_p, 3)
    result, _ = torch.sort(result, dim=-1, descending=False)
    return result
def GFM(src, tgt, Sharp=20, Threshold=0.5, k=10, eps=1e-6):
    index = torch.arange(src.shape[2]).reshape(1, -1, 1)
    index = index.repeat(src.shape[0], 1, k * (k - 1) // 2).unsqueeze(-1).cuda()
    idx = knn_tri(src, k)
    pairs = list(itertools.combinations(torch.arange(0, k), 2))
    idx_pairs = idx[:, :, pairs]
    src_T_pairs = torch.cat((index, idx_pairs), dim=-1)
    src = src.transpose(-1, -2)
    length_src = getTri(src, src_T_pairs)
    length_tgt = getTri(tgt, src_T_pairs) + eps
    loss = torch.sum((length_src - length_tgt) ** 2, dim=-1)
    loss = loss / torch.sum((length_src + length_tgt) ** 2, dim=-1)
    loss, _ = torch.sort(loss, dim=-1, descending=False)
    reg_loss = torch.norm(length_src - length_tgt, dim=-1)
    loss += 0.1 * reg_loss
    median_loss = torch.median(loss, dim=-1, keepdim=True)[0]
    outlier_mask = loss > 3 * median_loss
    loss[outlier_mask] = 0
    loss = loss[:, :, :k]
    loss = torch.sqrt(loss + eps)
    loss = torch.mean(loss, dim=-1)
    Min_loss, _ = torch.min(loss, dim=-1)
    Min_loss = Min_loss.reshape(-1, 1)
    loss = loss - Min_loss
    weight = 2 * torch.sigmoid(-Sharp * (loss.cuda()))
    weight[weight <= Threshold] = 0
    weight[weight > Threshold] = 1
    return weight

# 表示源点云中的点与目标点云中的点之间的匹配关系
def get_matching_indices(source, target, search_voxel_size, K=None):
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    match_inds = []
    for i, point in enumerate(source.points):
        # search_voxel_size: 表示搜索体素的大小，即在该体素内的点被视为匹配点
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def knn(x, k):
    x = x.transpose(1, 2)
    distance = -torch.cdist(x, x) ** 2
    distance += 1e-7
    idx = distance.topk(k=k, dim=-1)[1]
    return idx
def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)

class Generator(nn.Module):
    def __init__(self, n_emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(n_emb_dims, n_emb_dims // 2),
                                nn.LayerNorm(n_emb_dims // 2),
                                nn.ELU(),
                                nn.Linear(n_emb_dims // 2, n_emb_dims // 4),
                                nn.LayerNorm(n_emb_dims // 4),
                                nn.ELU(),
                                ),
        self.proj_rot = nn.Linear(n_emb_dims // 4, 4)
        self.proj_trans = nn.Linear(n_emb_dims // 4, 3)
    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = F.threshold(x, threshold=0.2, value=0., inplace=True)
        x1 = self.linears[-1](x)
        x1 = torch.mean(x1, dim=-1, keepdim=True)
        x = x + x1
        return x
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.encoder = Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N)
        self.decoder = Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N)
    def forward(self, src, tgt):
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        src_embedding = self.encoder(src, None)
        tgt_embedding = self.decoder(tgt, src_embedding, None, None)
        src_embedding = src_embedding.transpose(2, 1).contiguous()
        tgt_embedding = tgt_embedding.transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

# 不知道干什么的 知道返回了一个预测值和一个两个值之间的差
class TemperatureNet(nn.Module):
    def __init__(self, args):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.temp_factor = args.temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, 128),
                                nn.LayerNorm(128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.ELU(),
                                nn.Linear(128, 1),
                                nn.ELU())
        self.feature_disparity = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=2)
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding - tgt_embedding)

        self.feature_disparity = residual

        Temperature = torch.clamp(self.nn(residual), 1.0 / self.temp_factor, 1.0 * self.temp_factor)
        return Temperature, residual
def weighted_procrustes(X, Y, w, eps):
    assert len(X) == len(Y)
    W1 = torch.abs(w).sum()
    w = w.reshape(-1,1).repeat(1,3)
    w_norm = w / (W1 + eps)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
    U, D, V = Sxy.svd()
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1
    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    return R, t

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
                r, t = weighted_procrustes(src[i].cpu(), src_corr.cpu(), weight[i].detach().cpu(), 1e-7)
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
                r, t = weighted_procrustes(src[i].cpu(), src_corr.cpu(), weight[i].detach().cpu(), 1e-7)
                R.append(r)
                T.append(t)
            R = torch.stack(R, dim=0).cuda()
            T = torch.stack(T, dim=0).cuda()
            return R, T, corres, weight
class Position_encoding(nn.Module):
    def __init__(self, len, ratio):
        super(Position_encoding, self).__init__()
        self.PE = nn.Sequential(
            nn.Linear(3, len * ratio),
            nn.Sigmoid(),
            nn.Linear(len * ratio, len),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.PE(x)

class ChannelAttention(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(ch_in, ch_in // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_in // reduction, ch_in, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        return x * avg_out + x * max_out

class DualAttentionModule(nn.Module):
    def __init__(self, ch_in, reduction=16, num_heads=4):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(ch_in, reduction)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=ch_in, num_heads=num_heads)
    def forward(self, x):
        channel_output = self.channel_attention(x)
        x_reshaped = x.permute(2, 0, 1)
        spatial_output, _ = self.spatial_attention(x_reshaped, x_reshaped, x_reshaped)
        spatial_output = spatial_output.permute(1, 2, 0)
        output = channel_output * spatial_output
        return output


class T_prediction(nn.Module):
    def __init__(self, args):
        super(T_prediction, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.Position_encoding = Position_encoding(args.n_emb_dims, 8)
        self.SE_Block = DualAttentionModule(ch_in=args.n_emb_dims)
        self.emb_nn = GFM(embed_dim=args.n_emb_dims, token_dim=args.token_dim)
        self.attention = Transformer(args=args)
        self.temp_net = TemperatureNet(args)
        self.head = SVDHead(args=args)

    def forward(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr = self.predict_embedding(*input)
        rotation_ab, translation_ab, corres_ab, weight_ab = self.head(src_embedding, tgt_embedding, src, tgt,
                                                                      temperature, is_corr)
        rotation_ba, translation_ba, corres_ba, weight_ba = self.head(tgt_embedding, src_embedding, tgt, src,
                                                                      temperature, is_corr)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab
    # 生成嵌入向量
    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        is_corr = input[2]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        src_encoding = self.Position_encoding(src.transpose(1, 2)).transpose(1, 2).contiguous()
        tgt_encoding = self.Position_encoding(tgt.transpose(1, 2)).transpose(1, 2).contiguous()

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding + src_encoding, tgt_embedding + tgt_encoding)
        src_embedding = self.SE_Block(src_embedding + src_embedding_p)
        tgt_embedding = self.SE_Block(tgt_embedding + tgt_embedding_p)
        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr
    # 目标点预测，返回源、目标和得分
    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size * num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

# 创建点云对象
def make_open3d_point_cloud(xyz, color=None):
    # 就将点的坐标信息存储到了点云对象中
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # 颜色
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
            pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

# 预测建立的关系是否正确 没啥用啊
def corres_correct(matches, corres):
    """
    Function: verify that the correspondences established are correct
    Param:
        matches: ground-truth correspondences
        corres: predicted correspondences
    Return:
        is_correct: true if the correspondence is correct, otherwise, false
    """
    B, N, _ = corres.shape
    idx_src = torch.arange(N).reshape(1, -1, 1).repeat(B, 1, 1)
    corres = torch.cat([idx_src, corres], dim=-1).int()
    is_correct = []
    for match, corres_iter in zip(matches, corres):
        is_correct_iter = find_correct_correspondence(torch.tensor(match).int().unsqueeze(0), corres_iter.unsqueeze(0),
                                                      N)
        is_correct.extend(is_correct_iter)
    return np.array(is_correct)

# 找到正确的对应关系，以描述源点云和目标点云之间的几何变换
def find_matches(src, tgt, rotation, translation):
    """
    Function: find correct correspondences
    Param:
        src, tgt: point clouds [B, N, 3]
        rotatoin, translation: the ground-truth transformation
    Return:
        matches:  ground-truth correspondences
    """
    # 对输入的点云进行几何变换，根据给定的旋转和平移参数对点云进行相应的旋转和平移操作。
    src = apply_transformation_to_point_cloud(src, rotation, translation)
    matches = []
    for pointcloud1, pointcloud2 in zip(src, tgt):
        pcd0 = make_open3d_point_cloud(pointcloud1.cpu().numpy().T)
        pcd1 = make_open3d_point_cloud(pointcloud2.cpu().numpy().T)
        match = get_matching_indices(pcd0, pcd1, 0.04)
        matches.append(match)
    return matches

# 计算哈希值
def _hash(arr, M=None):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec

# 比较真实的对应关系和预测的对应关系，并确定哪些对应关系是正确的
def find_correct_correspondence(pos_pairs, pred_pairs, hash_seed=None):
    assert len(pos_pairs) == len(pred_pairs)


    corrects = []
    for i, pos_pred in enumerate(zip(pos_pairs, pred_pairs)):
        pos_pair, pred_pair = pos_pred
        if isinstance(pos_pair, torch.Tensor):
            pos_pair = pos_pair.numpy()
        if isinstance(pred_pair, torch.Tensor):
            pred_pair = pred_pair.numpy()

        _hash_seed = hash_seed

        pos_keys = _hash(pos_pair, _hash_seed)
        pred_keys = _hash(pred_pair, _hash_seed)

        corrects.append(np.isin(pred_keys, pos_keys, assume_unique=False))
    return np.hstack(corrects)

# 实现动态物体跟踪
class DIT(nn.Module):
    def __init__(self, args):
        super(DIT, self).__init__()
        self.num_iters = args.n_iters
        self.logger = Logger(args)
        self.discount_factor = args.discount_factor
        self.discrimination_loss = args.discrimination_loss
        self.T_prediction = T_prediction(args)
        self.model_path = args.model_path
        self.cycle_consistency_loss = args.cycle_consistency_loss
        self.crit = nn.BCELoss()

        if torch.cuda.device_count() > 1:
            self.T_prediction = nn.DataParallel(self.T_prediction)
    # 调用 T_prediction 模块对输入数据进行处理，返回预测的旋转、平移以及其他相关信息
    def forward(self, *input):
        rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab = self.T_prediction(
            *input)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab
    # 进行动态物体跟踪的预测，接受源点云 src 和目标点云 tgt，以及迭代次数 n_iters。
    # 在每次迭代中，通过调用 forward 方法预测源点云到目标点云的变换，并更新预测的旋转和平移参数
    def predict(self, src, tgt, n_iters=2):
        batch_size = src.size(0)
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        for i in range(n_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, _ \
                = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            src = apply_transformation_to_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        return rotation_ab_pred, translation_ab_pred
    def _train_one_batch(self, pcd, Transform, opt):
        opt.zero_grad()
        src, tgt, rotation_ab, translation_ab = pcd['src'], pcd['tgt'], Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        # 计算均方误差的矩阵
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        # 存储模型预测的旋转矩阵
        rotation_ab_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        # 存储模型预测的平移矩阵
        translation_ab_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}
        # 计算真实的匹配对应关系
        corres_gt = find_matches(src, tgt, rotation_ab, translation_ab)

        for i in range(self.num_iters):
            is_corr = 1
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
                feature_disparity, corres_ab, weight_ab = self.forward(src, tgt, is_corr)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i

            src = apply_transformation_to_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += loss
            is_correct = torch.tensor(corres_correct(corres_gt, corres_ab.cpu())).squeeze()
            accuracy = is_correct.sum() / is_correct.shape[0]
            if self.discrimination_loss != 0:
                discrimination_loss = self.discrimination_loss * self.crit((weight_ab).reshape(-1, 1).squeeze().cpu(),
                                                                           is_correct.to(
                                                                               torch.float)) * self.discount_factor ** i
                total_discrimination_loss += discrimination_loss
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor ** i
            scale_consensus_loss = 0
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + cycle_consistency_loss + scale_consensus_loss + discrimination_loss
        # 反向传播，计算梯度
        total_loss.backward()
        # 将参数朝着减少损失函数的方向更新
        opt.step()

        losses['total_loss'] = total_loss.item()
        losses['cycle'] = total_cycle_consistency_loss.item()
        losses['scale'] = total_scale_consensus_loss
        losses['acc'] = accuracy
        losses['discrimination'] = total_discrimination_loss.item()
        losses['mse'] = total_mse_loss.item()
        Transforms_Pred['R_ab_pred'] = rotation_ab_pred
        Transforms_Pred['T_ab_pred'] = translation_ab_pred

        return losses, Transforms_Pred


    def _test_one_batch(self, pcd, Transform, vis):
        src, tgt, rotation_ab, translation_ab = pcd['src'], pcd['tgt'], Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}

        corres_gt = find_matches(src, tgt, rotation_ab, translation_ab)
        if vis == True:
            self.Visulization(pcd)
        for i in range(self.num_iters):
            is_corr = (i != 0)
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
            feature_disparity, corres_ab, weight_ab = self.forward(src, tgt, is_corr)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i

            src = apply_transformation_to_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += loss
            discrimination_loss = 0
            is_correct = torch.tensor(corres_correct(corres_gt, corres_ab.cpu())).squeeze()
            accuracy = is_correct.sum() / is_correct.shape[0]
            if self.discrimination_loss != 0:
                discrimination_loss = self.discrimination_loss * self.crit((weight_ab).reshape(-1, 1).squeeze().cpu(),
                                                                           is_correct.to(
                                                                               torch.float)) * self.discount_factor ** i
                total_discrimination_loss += discrimination_loss
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor ** i
            scale_consensus_loss = 0
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + cycle_consistency_loss + scale_consensus_loss + discrimination_loss
            if vis == True:
                pcd['src'] = src
                self.Visulization(pcd)
            losses['total_loss'] = total_loss.item()
            losses['cycle'] = total_cycle_consistency_loss.item()
            losses['scale'] = total_scale_consensus_loss
            losses['acc'] = accuracy
            losses['discrimination'] = total_discrimination_loss.item()
            losses['mse'] = total_mse_loss.item()
            Transforms_Pred['R_ab_pred'] = rotation_ab_pred
            Transforms_Pred['T_ab_pred'] = translation_ab_pred
        return losses, Transforms_Pred
# 颜色
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
    # 一个Epoch就是将所有训练样本训练一次的过程

    # def _train_one_epoch(self, epoch, train_loader, opt, args):
    #     self.train()
    #     total_loss = 0
    #     rotations_ab = []
    #     translations_ab = []
    #     rotations_ab_pred = []
    #     translations_ab_pred = []
    #     eulers_ab = []
    #     num_examples = 0
    #     total_corres_accuracy = 0.0
    #     total_cycle_consistency_loss = 0.0
    #     total_scale_consensus_loss = 0.0
    #     total_mse_loss = 0.0
    #     total_discrimination_loss = 0.0
    #     avg_losses = {}
    #     Transforms = {}
    #     Metrics_mode = {}
    #     vis = False
    #     for pcd, Transform in tqdm(train_loader):
    #         for key in pcd.keys():
    #             pcd[key] = pcd[key].cuda()
    #         for key in Transform.keys():
    #             Transform[key] = Transform[key].cuda()
    #         losses, Transform_pred = self._train_one_batch(pcd, Transform, opt)
    #         batch_size = pcd['src'].size(0)
    #         num_examples += batch_size
    #
    #         total_mse_loss += losses['mse'] * batch_size
    #         total_loss = total_loss + losses['total_loss'] * batch_size
    #         total_cycle_consistency_loss = total_cycle_consistency_loss + losses['cycle'] * batch_size
    #         total_scale_consensus_loss = total_scale_consensus_loss + losses['scale'] * batch_size
    #         total_corres_accuracy += losses['acc'] * batch_size
    #         total_discrimination_loss += losses['discrimination'] * batch_size
    #
    #         rotations_ab.append(Transform['R_ab'].detach().cpu())
    #         translations_ab.append(Transform['T_ab'].detach().cpu())
    #         rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
    #         translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())
    #         eulers_ab.append(Transform['euler_ab'].cpu().numpy())
    #
    #     avg_losses['avg_loss'] = total_loss / num_examples
    #     avg_losses['avg_cycle'] = total_cycle_consistency_loss / num_examples
    #     avg_losses['avg_scale'] = total_scale_consensus_loss / num_examples
    #     avg_losses['avg_acc'] = total_corres_accuracy / num_examples
    #     avg_losses['avg_discrimination'] = total_discrimination_loss / num_examples
    #     avg_losses['avg_mse'] = total_mse_loss / num_examples
    #     avg_losses['epoch'] = epoch
    #     avg_losses['stage'] = 'train'
    #     Transforms['R_ab'] = torch.cat(rotations_ab, axis=0)
    #     Transforms['T_ab'] = torch.cat(translations_ab, axis=0)
    #     Transforms['R_ab_pred'] = torch.cat(rotations_ab_pred, axis=0)
    #     Transforms['T_ab_pred'] = torch.cat(translations_ab_pred, axis=0)
    #     Transforms['euler_ab'] = np.degrees(np.concatenate(eulers_ab, axis=0))
    #     Transforms['euler_ab_pred'] = npmat2euler(Transforms['R_ab_pred'])
    #
    #     Metrics_mode['cur_list'] = True
    #     Metrics_mode['save_mae'] = True
    #     Metrics_mode['outfile'] = 'mae.npy'
    #
    #     return self.Compute_metrics(avg_losses, Transforms, Metrics_mode), avg_losses['avg_acc']
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
        self.path = 'checkpoints/' + args.exp_name
        self.fw = open(self.path + '/log', 'a')
        self.fw.write(str(args))
        self.fw.write('\n')
        self.fw.flush()
        print(str(args))
        with open(os.path.join(self.path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def write(self, info):
        arrow = info['arrow']
        epoch = info['epoch']
        stage = info['stage']
        loss = info['loss']
        cycle_consistency_loss = info['cycle_consistency_loss']
        scale_consensus_loss = info['scale_consensus_loss']
        mse_loss = info['mse_loss']
        discrimination_loss = info['dis_loss']
        r_ab_mse = info['r_ab_mse']
        r_ab_rmse = info['r_ab_rmse']
        r_ab_mae = info['r_ab_mae']
        t_ab_mse = info['t_ab_mse']
        t_ab_rmse = info['t_ab_rmse']
        t_ab_mae = info['t_ab_mae']
        r_ab_r2_score = info['r_ab_r2_score']
        t_ab_r2_score = info['t_ab_r2_score']
        r_ab_mie = info['r_ab_mie']
        t_ab_mie = info['t_ab_mie']
        corres_accuracy = info['corres_accuracy']
        cur_acc = info['cur_acc']
        text = '%s:: Stage: %s, Epoch: %d, Loss: %f, Cycle_consistency_loss: %f, ' \
               'Scale_consensus_loss: %f, dis_loss: %f, mse_loss: %f, Rot_MSE: %.10f, Rot_RMSE: %.10f, ' \
               'Rot_MAE: %.10f, Rot_R2: %f, Trans_MSE: %.10f, ' \
               'Trans_RMSE: %.10f, Trans_MAE: %8.10f, Trans_R2: %f, ' \
               'corres_accuracy: %f, r_ab_mie: %.10f, t_ab_mie: %.10f, cur_acc:%f\n' % \
               (arrow, stage, epoch, loss, cycle_consistency_loss, scale_consensus_loss,
                discrimination_loss, mse_loss, r_ab_mse, r_ab_rmse, r_ab_mae,
                r_ab_r2_score, t_ab_mse, t_ab_rmse, t_ab_mae, t_ab_r2_score, corres_accuracy, r_ab_mie, t_ab_mie,
                cur_acc)
        self.fw.write(text)
        self.fw.flush()
        print(text)

    def close(self):
        self.fw.close()


if __name__ == '__main__':
    print('hello world')