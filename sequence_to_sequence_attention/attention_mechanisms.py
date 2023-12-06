from __future__ import unicode_literals, print_function, division

import math

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import seaborn as sns

#To ensure reproducibility of the experiments, we can set the seed to a fixed number.
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)
_ = torch.mps.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  device = torch.device("mps")

print(device)


def plot_attention(attention, xtitle="Keys", ytitle="Queries"):
  """ Plots the attention map."""

  sns.set(rc={'figure.figsize': (12, 8)})
  ax = sns.heatmap(
    attention.detach().cpu(),
    linewidth=0.5,
    cmap="Blues",
    square=True)

  ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
  ax.set_xlabel(xtitle)
  ax.set_ylabel(ytitle)

  plt.show()

def dummy_train(attn_module, q, k, v, target_attn_weights):
  optimizer = optim.Adam(attn_module.parameters(), lr=0.0001)

  target = torch.matmul(target_attn_weights, v)

  attn_module.train()

  n_epochs = 15000
  for i in range(n_epochs):
    optimizer.zero_grad()
    output, attn_weights = attn_module(q, k, v)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    if (i + 1) % 1000 == 0:
      print(f"Epoch {i + 1}/{n_epochs}")
      print(f"\tLoss:\t\t{loss.item()}")
      print(f"\tAttention:\t{attn_weights.squeeze().detach().numpy().round(3)}")

  attn_module.eval()
  output, attn_weights = attn_module(q, k, v)

  print(f"\nOutput:\n{output}\n")
  print(f"\nTarget:\n{target}\n")

  return output, attn_weights

class MultiplicativeAttention(nn.Module):
  """
   Implements plain dot-product and multiplicative attention.
   Args:
       q_dim (int): dimension of the queries
       k_dim (int): dimension of the keys
       v_dim (int): dimension of the values
       scaling (bool): whether to scale after the dot-product
       sub_type (string): specify type of attention: dot-product / multiplicative
   Inputs: query, key, value
      query (torch.FloatTensor):  Query Tensor   (... x T_q x d_q)
      key (torch.FloatTensor):  Key Tensor     (... x T_k x d_k)
      value (torch.FloatTensor):  Value Tensor   (... x T_v x d_v)
  Returns:
      torch.FloatTensor: Result of the Attention Mechanism  (... x T_q x d_v)
      torch.FloatTensor: Attention map       (... x T_q x T_k)
  """
  def __init__(self, q_dim: int, k_dim: int, scaling: bool = False, sub_type: str = 'multiplicative') -> None:
    super(MultiplicativeAttention, self).__init__()
    self.sub_type = sub_type
    if self.sub_type == 'dot_product':
      assert q_dim == k_dim
      self.scaling = scaling
    else:
      self.proj_w = nn.Linear(q_dim, k_dim, bias=False)
      self.scaling = False  # We don't scale in the multiplicative attention
  def forward(self, query, key, value):
    if self.sub_type == 'dot_product':
      qw = query
      key_ = key
    else:
      qw = self.proj_w(query)  # (... x T_q x q_dim) -> (... x T_q x k_dim)
      key_ = key
    # TODO: Compute the attention logits from qw and the key
    #  (... x T_q x k_dim) * (... x k_dim x T_k) -> (... x T_q x T_k)
    attn_logits = torch.matmul(qw, key_.transpose(-2, -1))
    if self.scaling:
      attn_logits = attn_logits / math.sqrt(key_.size(-1))
    # Before softmax the inputs are called always logits
    # TODO: Compute the attention weights
    attn_weights = F.softmax(input=attn_logits, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights

dim = 128
q = torch.randn(1, dim)
k = torch.randn(8, dim)
v = k

target_attn_weights = torch.Tensor([[0.1, 0.2, 0.1, 0.3, 0.0, 0.2, 0.0, 0.1]])

attn_module = MultiplicativeAttention(q_dim=dim, k_dim=dim)
output, attn_weights = dummy_train(attn_module, q, k, v, target_attn_weights)

# Plot attention weights of trained attention

output, attention = attn_module(q,k,v)

plot_attention(
    attention,
)

attn_module = MultiplicativeAttention(q_dim=dim, k_dim=dim, sub_type='dot_product')
output, attention_no_scaling = attn_module(q,k,v)

plot_attention(
    attention_no_scaling,
)

attn_module = MultiplicativeAttention(q_dim=dim, k_dim=dim, sub_type='dot_product', scaling = True)
output, attention_scaling = attn_module(q,k,v)

plot_attention(
    attention_scaling,
)


def plot_histogram(input_tensor_dict):
  """Helper function to make bar plots."""
  for key in input_tensor_dict.keys():
    input_tensor = input_tensor_dict[key].squeeze().cpu().detach().numpy()
    plt.bar(range(0, input_tensor.size), input_tensor, alpha=0.6, label=key)
  plt.xticks(ticks=range(0, input_tensor.size))
  plt.legend()
  plt.show()


attn_results_dict = {'No scaling': attention_no_scaling, 'Scaling': attention_scaling}
plot_histogram(attn_results_dict)


"""
Additive Attention
"""


class AdditiveAttention(nn.Module):
  """
   Implements the additive attention as proposed in "Neural Machine Translation by Jointly Learning to Align and Translate".
   Args:
       q_dim (int): dimesion of the queries
       k_dim (int): dimesion of the keys
       attn_dim (int): dimension of intermediate vectors

   Inputs: query, key, value
      query (torch.FloatTensor):  Query Tensor   (... x T_q x d_q)
      key (torch.FloatTensor):  Key Tensor     (... x T_k x d_k)
      value (torch.FloatTensor):  Value Tensor   (... x T_v x d_v)

  Returns:
      torch.FloatTensor: Result of the Attention Mechanism  (... x T_q x d_v)
      torch.FloatTensor: Attention map       (... x T_q x T_k)

  """

  def __init__(self, q_dim: int, k_dim: int, attn_dim: int) -> None:
    super(AdditiveAttention, self).__init__()

    # TODO: Create projections of queries and keys
    self.proj_q = nn.Linear(q_dim, attn_dim, bias=False)
    self.proj_k = nn.Linear(k_dim, attn_dim, bias=False)

    self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
    self.w = nn.Linear(attn_dim, 1)

  def forward(self, query, key, value):
    q_ = self.proj_q(query)  # (... x T_q x q_dim) -> (... x T_q x attn_dim)
    k_ = self.proj_k(key)  # (... x T_k x k_dim) -> (... x T_k x attn_dim)

    # Prepare for Broadcasting Semantics
    q_ = q_.unsqueeze(-2)  # (... x T_q x attn_dim) -> (... x T_q x  1  x attn_dim)
    k_ = k_.unsqueeze(-3)  # (... x T_k x attn_dim) -> (... x  1  x T_k x attn_dim)

    #  Sum thanks to Broadcasting Semantics
    attn_hid = torch.tanh(
      q_ + k_ + self.bias)  # (... x T_q x  1  x attn_dim) + (... x  1  x T_k x attn_dim) + (attn_dim) -> (... x T_q x T_k x attn_dim)

    attn_logits = self.w(attn_hid)  #  (... x T_q x T_k x attn_dim) -> (... x T_q x T_k x 1)
    attn_logits = attn_logits.squeeze(-1)  # (... x T_q x T_k x 1) -> (... x T_q x T_k)

    attn_weights = F.softmax(attn_logits, dim=-1)

    # TODO: Compute the output of the attention
    output = torch.matmul(attn_weights, value)

    return output, attn_weights

target_attn_weights = torch.Tensor([[0.1, 0.2, 0.1, 0.3, 0.0, 0.2, 0.0, 0.1]])

attn_module = AdditiveAttention(q_dim=dim, k_dim=dim, attn_dim=dim//2)

output, attn_weights = dummy_train(attn_module, q, k, v, target_attn_weights)

# Plot attention weights of trained attention
output, attention = attn_module(q,k,v)

plot_attention(
    attention,
)

"""
Attention with matrices
"""

dim = 128
q = torch.randn(3, dim)
k = torch.randn(8, dim)
v = k

target_attn_weights = torch.Tensor([[0.1, 0.2, 0.1, 0.3, 0.0, 0.2, 0.0, 0.1],
                                    [0.4, 0.1, 0.1, 0.2, 0.0, 0.2, 0.0, 0.0],
                                    [0.0, 0.6, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1]])

attn_module = AdditiveAttention(q_dim=dim, k_dim=dim, attn_dim=dim//2)

output, attn_weights = dummy_train(attn_module, q, k, v, target_attn_weights)

output, attention = attn_module(q,k,v)

plot_attention(
    attention,
)