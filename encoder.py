"""Local graph encoder and task-local representation extraction."""
import torch
from torch import nn
from torch.nn import functional as F

# encoder
class LocalGraphBlock(nn.Module):
    def __init__(self,dimension,heads,dropout):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(dimension,heads,dropout=dropout,batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dimension,2*dimension),nn.GELU(),
                                 nn.Dropout(dropout),nn.Linear(2*dimension,dimension))
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self,h,adjacency):
        valid = adjacency>=0
        neighbors = h[adjacency.clamp_min(0)]
        center = h[:,None,:]
        values = torch.cat((center,neighbors),dim=1)
        padding = torch.cat((torch.zeros((len(h),1),device=h.device,dtype=torch.bool),~valid),dim=1)
        local,_ = self.local_attn(center,values,values,key_padding_mask=padding,need_weights=False)
        h = self.norm1(h+self.dropout(local[:,0]))
        return self.norm2(h+self.dropout(self.ffn(h)))


class LocalGraphEncoder(nn.Module):
    def __init__(self,feature_dim,hidden_dim=256,dropout=.1,heads=4,layers=2):
        super().__init__()
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim,hidden_dim),nn.LayerNorm(hidden_dim),
                                             nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([LocalGraphBlock(hidden_dim,heads,dropout) for _ in range(layers)])
        self.readout_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim),
                                                        nn.Linear(hidden_dim,hidden_dim),nn.Tanh())])

    def forward(self,features,adjacency):
        h = self.dropout(self.layer_norm_input(self.feature_encoder(features)))
        for layer in self.layers:
            h = layer(h,adjacency)
        return F.normalize(self.readout_heads[0](h),dim=-1)

# local views
def local_batch(x, padded, support, query):
    ids = torch.cat((support, query)).to(x.device)
    first = padded[ids].reshape(-1)
    nodes = torch.cat((ids, first[first >= 0])).unique(sorted=True)
    neighbors = padded[nodes]
    valid = neighbors >= 0
    positions = torch.searchsorted(nodes, neighbors.clamp_min(0)).clamp_max(len(nodes) - 1)
    valid &= nodes[positions] == neighbors
    adjacency = torch.where(valid, positions, torch.full_like(positions, -1))
    centered = x[nodes] - x[nodes].mean(0, keepdim=True)
    return centered, adjacency, torch.searchsorted(nodes, ids)

def encode(backbone, features, adjacency, ids):
    latent = backbone(features,adjacency)[ids]
    valid = adjacency >= 0
    smoothed = (features + (features[adjacency.clamp_min(0)] * valid[..., None]).sum(1)) / (1 + valid.sum(-1))[:, None]
    return F.normalize(features, dim=-1)[ids], F.normalize(smoothed, dim=-1)[ids], latent
