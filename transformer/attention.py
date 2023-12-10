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

# Test SDPA function.

q = torch.randn(5, 4)
k = torch.randn(8, 4)
v = torch.randn(8, 4)

output, attention = scaled_dot_product(q, k, v)

print(f"Output:\n{output}\n{output.shape}\n")
print(f"Attention weights:\n{attention}\n{attention.shape}\n")

plot_attention(
    attention,
    [str([round(float(q__), 1) for q__ in q_]) for q_ in q],
    [str([round(float(k__), 1) for k__ in k_]) for k_ in k],
)

# Naive self-attention
x = torch.randn(5, 4)
output, attention = scaled_dot_product(q=x, k=x, v=x)

print(f"Output:\n{output}\n{output.shape}\n")
print(f"Attention weights:\n{attention}\n{attention.shape}\n")

plot_attention(
    attention,
    [str([round(float(q__), 1) for q__ in q_]) for q_ in q],
    [str([round(float(q__), 1) for q__ in q_]) for q_ in q],
)

# We need learnable parameters
# Class implementation
class LearnableScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LearnableScaledDotProductAttention, self).__init__()
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        output, _ = scaled_dot_product(q, k, v, mask)
        return output

sdpa = LearnableScaledDotProductAttention(embed_dim=4)
optimizer = optim.Adam(sdpa.parameters())

losses_sdpa = []
n_epochs = 10000
for i in range(n_epochs):
    optimizer.zero_grad()
    output = sdpa(q=x, k=x, v=x)    # Self-attention
    loss = F.mse_loss(output, x)    # Reconstruct the input
    loss.backward()
    optimizer.step()
    losses_sdpa.append(loss.item())
    if (i + 1) % 1000 == 0:
        print(f"Loss ({i+1}/{n_epochs}): {loss.item()}")


print(f"\nOutput:\n{output}\n")
print(f"Query:\n{x}\n")
