import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value, attn_mask=None):
        # Cross attention expects query, key, and value as (batch, seq_len, embed_dim)
        attn_output, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        return attn_output
