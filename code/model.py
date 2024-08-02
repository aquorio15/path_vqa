from layer import *
from attention import *
from encoder_block import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import List
from fairseq.modules import (
    FairseqDropout,
    PositionalEmbedding,
)
from fairseq.data.data_utils import lengths_to_padding_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

class Conv1dSubsampler(nn.Module):

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

class E2E_MVQA(nn.Module):
    
    def __init__(
        self,
        input_dim_1: int,
        input_dim_2: int,
        input_dim_3: int,
        feat_dim: int,
        conv_channels: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        n_heads: int,
        encoder_layers: int,
        dropout: int=0.1,
        num_classes: int=1000, 
        kernel: List[int] = (5, 5),
        encoder_normalize_before: bool=True,
        max_seq_len: int=1000
    ):
        
        super(E2E_MVQA, self).__init__()

        self.transformer_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_heads, dropout=dropout, batch_first=True, dim_feedforward=d_ff) 
             for _ in range(encoder_layers)]
        )
        self.subsample1 = Conv1dSubsampler(
                input_dim_1,
                conv_channels,
                feat_dim,
                [int(k) for k in kernel.split(",")],
            )
        self.subsample2 = Conv1dSubsampler(
                input_dim_2,
                conv_channels,
                feat_dim,
                [int(k) for k in kernel.split(",")],
            )
        self.subsample3 = Conv1dSubsampler(
                input_dim_3,
                conv_channels,
                feat_dim,
                [int(k) for k in kernel.split(",")],
            )
        self.dropout_module = FairseqDropout(
            p=dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(feat_dim)
        self.padding_idx = 1
        self.embed_positions = PositionalEmbedding(
            max_seq_len, feat_dim, self.padding_idx
        )
        if encoder_normalize_before:
            self.layer_norm = nn.LayerNorm(feat_dim)
        else:
            self.layer_norm = None
        self.mlp = nn.Linear(feat_dim, num_classes)
        self.image = nn.Linear(input_dim_2, feat_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.last = nn.Linear(15168, feat_dim)
    
    def get_atten_mask(self, seq_lens, batch_size):
        max_len = seq_lens[0]
        atten_mask = torch.ones([batch_size, max_len, max_len])
        for i in range(batch_size):
            length = seq_lens[i]
            atten_mask[i, :length,:length] = 0
        return atten_mask.bool()
    
    def forward(self, src_token, src_length, image_token, tgt_length, txt_token, text_length, attention_mask=None):
        b, s, d = src_token.size(0), src_token.size(1), src_token.size(2)
        x, input_lengths1 = self.subsample1(src_token, src_length)
        y, input_lengths2 = self.subsample2(image_token, tgt_length)
        a, input_lengths3 = self.subsample2(txt_token, text_length)
        
        # y = self.image(image_token)
        y = y.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        # z = torch.cat([x, y], dim=1)
        z = x * y
        z = self.embed_scale * z
        padding_length = torch.tensor([z.shape[1]]).cuda()
        encoder_padding_mask = lengths_to_padding_mask(padding_length)
        positions = self.embed_positions(encoder_padding_mask)
        z += positions
        z = self.dropout_module(z)
        # if attention_mask is not None:
        #     # Create square attention mask
        #     attention_mask = nn.Transformer.generate_square_subsequent_mask(len(z)).to(device)
        for layer in self.transformer_layers:
            z = layer(z)
        # if self.layer_norm is not None:
            # z = self.layer_norm(z)
        z = z.mean(dim=1)    
        # z = z[:, -1, :].view(b, -1)
        z = self.mlp(z)
        return z

# model = E2E_MVQA(input_dim=768,
#         feat_dim=64,
#         conv_channels=1024,
#         image_feat_len=1024,
#         d_k=128,
#         d_v=128,
#         d_ff=1024,
#         n_heads=8,
#         kernel="3,3,3,3",
#         encoder_layers=6,
#         dropout=0.2,
#         num_classes=1000, 
#         encoder_normalize_before=True,
#         max_seq_len=1000).cuda()

# # # # NOTE: Check the model
# src_token = torch.randn(32, 543, 768).cuda()
# src_lengths = torch.Tensor([src_token.shape[1]]).long().cuda()
# image_token = torch.randn(32, 49, 1024).cuda()
# print(model(src_token, src_lengths, image_token))
