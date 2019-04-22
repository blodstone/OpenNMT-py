"""Topic attention modules (Hardy)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicAttention(nn.Module):
    """
    Topic attention takes an LDA matrix, encoder matrix and a query vector.
    It then computes a parameterized convex combination of the matrix based on
    the input query
    """

    def __init__(self, dim, attn_type="dot", attn_func="softmax"):
        super(TopicAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

    def score(self, h_t, h_s, h_k):
        pass