from layer import *
from attention import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, feat_dim, d_k, d_v, d_ff, n_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(feat_dim, d_k, d_v, n_heads, dropout)
        self.position_wise_ff = PositionWiseFeedForward(feat_dim, d_ff, dropout)
    def forward(self, x, atten_mask):
        enc_output, atten = self.self_attention(x, atten_mask)
        enc_output = self.position_wise_ff(enc_output)
        return enc_output, atten