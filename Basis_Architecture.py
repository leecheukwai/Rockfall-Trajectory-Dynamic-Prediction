import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from TCN import ConvPyramidTCN
from Dataset_maker2 import read_Timewin,Tesdata
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
device = torch.device('cuda:0')

def evaluate(model, X_test, coords_test, y_test,batch_size=10):
    model.eval()
    correct = 0
    total = 0


    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            x_batch = X_test[i:i+batch_size]      # [B, N, C, T]
            c_batch = coords_test[i:i+batch_size] # [B, N, 2]
            y_batch = y_test[i:i+batch_size]      # [B,]

            logits = model(x_batch, c_batch)      # [B, num_classes]
            #Tloss=loss_function(logits,label_test)
            pred = torch.argmax(logits, dim=1)    # [B,]
            y_batch = torch.argmax(y_batch, dim=1)

            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    return acc

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=6, dropout=0.25):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.FC = nn.Sequential(
            nn.Linear(600, 128),#6=coordinate
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.transformer_encoder(x)
        pooled_feature = x.max(dim=2)[0]
        x=self.FC(pooled_feature)
        #x = x.permute(0, 2, 1)
        return x
def knn(x, k):
    """
    Compute k-nearest neighbors for each node based on distances.
    x: Tensor of shape [B, N, D]
    returns idx: LongTensor of shape [B, N, k]
    """
    dist = torch.cdist(x, x)  # [B, N, N]
    # Find k+1 nearest (including self) along last dimension
    _, idx_full = dist.topk(k + 1, dim=-1, largest=False)
    # Exclude self (first neighbor) to get exactly k neighbors
    neigh_idx = idx_full[:, :, 1:k + 1]
    return neigh_idx


class TemporalExtractor(nn.Module):
    def __init__(self, in_channels=3, out_dim=64, num_layers=7, kernel_size=4, pool_kernel=2,dropout=0.25):
        super().__init__()
        layers = []
        c = in_channels
        for i in range(num_layers):
            layers.append(nn.Conv1d(c, out_dim, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.MaxPool1d(pool_kernel))
            layers.append(nn.Dropout(dropout))
            c = out_dim
        # ensure single time output
        #layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)
        self.Trans=TransformerEncoder(d_model=3, nhead=3, num_layers=2, dropout=0.25)
        #self.Bl=BiLSTM(600)
        self.Tcn=ConvPyramidTCN(in_channels=3,
                           channel_sizes=[64,64,64, 64,64, 64, 64,64],
                           kernel_size=3,
                           dropout=0.25,
                           pyramid_kernel=3)

    def forward(self, x):
        # x: [B, N, C, T]
        B, N, C, T = x.shape
        x1 = x.contiguous().view(B * N, C, T)
        out=out.mean(-1)
        out=self.Tcn(x1)
        #out=self.Trans(x1)
        out = out.squeeze(-1)  # [B*N, D]
        return out.contiguous().view(B, N, -1)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3=GCNConv(hidden_channels,hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.drop=nn.Dropout(0.25)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x=self.drop(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.drop(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        return x

from torch_geometric.data import Data, Batch

class GCN_layers(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.indim = in_dim
        self.outdim = out_dim
        self.GCN = GCN(in_channels=in_dim, hidden_channels=in_dim, out_channels=out_dim)

    def fully_connected_edge_index(self, K, B, device):
        row = torch.arange(K, device=device).repeat(K)
        col = torch.arange(K, device=device).repeat_interleave(K)
        edge_index = torch.stack([row, col], dim=0)
        mask = row != col
        edge_index = edge_index[:, mask]  # [2, E]

        E = edge_index.shape[1]
        edge_index = edge_index.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, E]
        offset = (torch.arange(B, device=device) * K).view(B, 1, 1)  # [B,1,1]
        edge_index = edge_index + offset  # [B, 2, E]
        edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)  # [2, B*E]
        return edge_index  # [2, B*E]

    def compute_edge_weight(self, pos, edge_index, K, kernel='inverse', sigma=1.0):
        B, K, _ = pos.shape

        base_edge_index = self.fully_connected_edge_index(K, 1, pos.device)
        src = pos[:, base_edge_index[0]]  # [B, E, 3]
        dst = pos[:, base_edge_index[1]]  # [B, E, 3]
        dist2 = ((src - dst) ** 2).sum(dim=-1)  # [B, E]

        if kernel == 'inverse':
            ew = 1.0 / (dist2.sqrt() + 1e-6)
        elif kernel == 'gaussian':
            ew = torch.exp(-dist2 / (2 * sigma ** 2))
        else:
            raise ValueError("Unknown kernel")

        return ew.reshape(-1)  # [B*E]

    def build_graph_batch(self, feat, pos, kernel='inverse'):
        B, K, T = feat.shape
        device = feat.device

        edge_index = self.fully_connected_edge_index(K, B, device)  # [2, B*E]

        edge_weight = self.compute_edge_weight(pos, edge_index, K, kernel=kernel)  # [B*E]

        x = feat.reshape(B * K, T)   # [B*K, T]
        pos = pos.reshape(B * K, -1) # [B*K, 3]


        batch = torch.arange(B, device=device).repeat_interleave(K)  # [B*K]


        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch
        )
        return data

    def forward(self, feat, pos):
        B, K, T = feat.shape
        batch_data = self.build_graph_batch(feat, pos)
        out = self.GCN(batch_data.x, batch_data.edge_index, batch_data.edge_weight)
        out = out.view(B, K, -1)
        return out


class EdgeConv_D(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim*2+6, hidden_dim),#6=coordinate
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.gcn=GCN_layers(in_dim*2,in_dim)

    def forward(self, x, knn_idx,coord):
        # x: [B, N, D]
        B, N, D = x.shape
        Bc,Nc,Dc=coord.shape
        k = knn_idx.size(-1)
        idx = knn_idx.unsqueeze(-1).expand(B, N, k, D)
        idxc=knn_idx.unsqueeze(-1).expand(Bc, Nc, k, Dc)
        neigh = torch.gather(x.unsqueeze(1).expand(B, N, N, D), 2, idx)
        neighcoord=torch.gather(coord.unsqueeze(1).expand(Bc, Nc, Nc, Dc), 2, idxc)
        xi = x.unsqueeze(2).expand(B, N, k, D)
        coordi=coord.unsqueeze(2).expand(Bc, Nc, k, Dc)
        feat = torch.cat([xi,neigh-xi,coordi,neighcoord], dim=-1)
        out=self.gcn(feat,coordi)
        out=out.view(B, N, k, -1)
        return out.max(dim=2)[0]

class EdgeConv_F(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, hidden_dim)
        )
        self.gcn=GCN_layers(in_dim*2,in_dim)

    def forward(self, x, knn_idx):
        # x: [B, N, D]
        B, N, D = x.shape
        k = knn_idx.size(-1)
        idx = knn_idx.unsqueeze(-1).expand(B, N, k, D)
        neigh = torch.gather(x.unsqueeze(1).expand(B, N, N, D), 2, idx)
        xi = x.unsqueeze(2).expand(B, N, k, D)
        feat = torch.cat([xi,neigh-xi], dim=-1)#neigh-xi
        out = self.mlp(feat)
        return out.max(dim=2)[0]


class MGN(nn.Module):
    def __init__(self, C=3, T=4096, D=64, k=5, num_gc_layers=4, pred_dim=64):
        super().__init__()
        self.temporal = TemporalExtractor(in_channels=C, out_dim=64)
        self.dist_layers = nn.ModuleList([EdgeConv_D(D, D) for _ in range(num_gc_layers)])
        self.feat_layers = nn.ModuleList([EdgeConv_F(D, D) for _ in range(num_gc_layers)])
        self.k = k
        self.site_fc = nn.Linear((num_gc_layers+1)*D, pred_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pred_dim, pred_dim),#32
            nn.LeakyReLU(),
            nn.Linear(pred_dim, 3),
        )
        self.gcn = GCN_layers(64, 3)

    def forward(self, X, coords):
        # X: [B, N, C, T], coords: [B, N, 2]

        t0 = self.temporal(X)
        # [B, N, D]
        xs = [t0]
        x = t0

        for gc1,gc2 in zip(self.dist_layers,self.feat_layers):
            dist_idx = knn(coords, self.k)
            feat_idx = knn(x, self.k)
            x =gc2(x, feat_idx)+ gc1(x, dist_idx,coords)
            xs.append(x)

        x_cat = torch.cat(xs, dim=-1)
        x_site = self.site_fc(x_cat)
        x_pool, _ = torch.max(x_site, dim=1)
        out = self.classifier(x_pool)
        return out

criterion = nn.CrossEntropyLoss()
def loss_function(recon_x, x):
    AE=criterion(recon_x,x)
    return AE
def train():

    batch_size = 100
    device = torch.device('cuda:0')
    B, N, C, T = 20, 19, 3, 600
    model = MGN(C=C, T=T, D=64, k=8, num_gc_layers=4, pred_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)



if __name__ == "__main__":
    pass



