import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import math
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def scaled_dot_product(q, k, v, mask=None):
    """ Computes the Scaled Dot-Product Attention

    Args:
        q (torch.FloatTensor):  Query Tensor   (... x T_q x d_q)
        k (torch.FloatTensor):  Key Tensor     (... x T_k x d_k)
        v (torch.FloatTensor):  Value Tensor   (... x T_v x d_v)
        mask (torch.BoolTensor): Attention mask (... x T_q x T_k)

    Returns:
        torch.FloatTensor: Result of the SDPA  (... x T_q x d_v)
        torch.FloatTensor: Attention map       (... x T_q x T_k)

    """
    assert q.size(-1) == k.size(-1), "Query and Key dimensions must coincide"

    # TODO: Matrix multiplication of the queries and the keys (use torch.matmul)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))

    # TODO: Scale attn_logits (see the SDPA formula, d_k is the last dim of k)
    attn_logits = attn_logits / torch.sqrt(torch.tensor(k.size(-1)))

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, -float("inf"))

    # TODO: Compute the attention weights (see the SDPA formula, use dim=-1)
    softmax = nn.Softmax(dim=-1)
    attention = softmax(attn_logits)

    output = torch.matmul(attention, v)

    return output, attention

def plot_attention(attention, queries, keys, xtitle="Keys", ytitle="Queries"):
    """ Plots the attention map

    Args:
        att (torch.FloatTensor): Attention map (T_q x T_k)
        queries (List[str]): Query Tensor
        keys (List[str]): Key Tensor
    """

    sns.set(rc={'figure.figsize':(12, 8)})
    ax = sns.heatmap(
        attention.detach().cpu(),
        linewidth=0.5,
        xticklabels=keys,
        yticklabels=queries,
        cmap="coolwarm")

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    plt.show()

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be multiple of the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_o = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_o.weight)
        self.proj_q.bias.data.fill_(0)
        self.proj_k.bias.data.fill_(0)
        self.proj_v.bias.data.fill_(0)
        self.proj_o.bias.data.fill_(0)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(1)

        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # TODO: Split the tensors into multiple heads
        #  T x B x embed_dim -> T x B x num_heads x head_dim
        # -1 todo lo que queda
        q = q.reshape(-1, batch_size, self.num_heads, self.head_dim)
        k = k.reshape(-1, batch_size, self.num_heads, self.head_dim)
        v = v.reshape(-1, batch_size, self.num_heads, self.head_dim)

        # The last two dimensions must be sequence length and the head dimension,
        # to make it work with the scaled dot-product function.
        # TODO: Rearrange the dimensions
        # T x B x num_heads x head_dim -> B x num_heads x T x head_dim
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Apply the same mask to all the heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # TODO: Call the scaled dot-product function (remember to pass the mask!)
        output_heads, attn_w = scaled_dot_product(q, k, v, mask)

        # B x num_heads x T x head_dim -> T x B x num_heads x head_dim
        output_heads = output_heads.permute(2, 0, 1, 3)

        # T x B x num_heads x head_dim -> T x B x embed_dim
        output_cat = output_heads.reshape(-1, batch_size, self.embed_dim)
        output = self.proj_o(output_cat)

        return output, attn_w

mha = MultiheadAttention(embed_dim=4, num_heads=2)
optimizer = optim.Adam(mha.parameters())

x = torch.randn(5, 4)

losses_mha = []
n_epochs = 10000
for i in range(n_epochs):
    optimizer.zero_grad()
    output = mha(                # Self-attention
        q=x.unsqueeze(1),
        k=x.unsqueeze(1),
        v=x.unsqueeze(1)
    )[0].squeeze(1)
    loss = F.mse_loss(output, x) # Reconstruct input
    loss.backward()
    optimizer.step()
    losses_mha.append(loss.item())
    if (i + 1) % 1000 == 0:
        print(f"Loss ({i+1}/{n_epochs}): {loss.item()}")

print(f"\nOutput:\n{output}\n")
print(f"Query:\n{x}\n")