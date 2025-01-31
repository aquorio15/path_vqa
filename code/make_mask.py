import torch


# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0)