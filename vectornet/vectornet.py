import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_min, scatter_mean
import numpy as np
from vn_config import vn_config


class SubGraph(nn.Module):
    
#     Local Interaction Block

    def __init__(self, input = 64, hidden = 64): # 64 - value from the paper
        super(SubGraph, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
    
    
    def forward(self, X, i):
        fX = self.nn(X)
#         fX = fX / fX.norm(dim=0)
        SfX = scatter_max(fX, i, dim = 0)[0] #m
        res = torch.cat([fX, SfX[i.squeeze()]], dim = 1)
        
        return res


    
class SelfAttentionMapping(nn.Module):
    
    def __init__(self, input, hidden):
        super(SelfAttentionMapping, self).__init__()
        
        self.l = nn.Sequential(
            nn.Linear(input, hidden),
#             nn.BatchNorm1d(hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
        )
    
    
    def forward(self, x):
        return self.l(x)



class GlobalGraph(nn.Module):
    
    def __init__(self, input=32, hidden = 16):
        super(GlobalGraph, self).__init__()

        self.v = SelfAttentionMapping(input, hidden)
        self.q = SelfAttentionMapping(input, hidden)
        self.k = SelfAttentionMapping(input, hidden)

        
    def forward(self, x, batches, ego_idx):
        ego_idx = ego_idx.long()# ego_idx is always = 0 now
        values = []
        attentions = []
        
        Q = self.q(x)
        V = self.v(x)
        K = self.k(x)
        
        _Q = []
        _V = []
        _K = []
        QKt = []
        
        for b in range(batches.max().item() + 1):
            _Q.append(Q[batches == b])
            _V.append(V[batches == b])
            _K.append(K[batches == b])
            
            QKt.append(torch.softmax(torch.matmul(_Q[-1], _K[-1].permute(1, 0)) / np.sqrt(vn_config["hidden_dim"]), dim = 1))
            values.append(torch.matmul(QKt[-1], _V[-1])[0][None,])
            attentions.append(QKt[-1][0])
        
        return torch.cat(values, dim = 0), attentions