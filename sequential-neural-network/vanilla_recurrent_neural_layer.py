import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

torch.manual_seed(1)
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
  torch.cuda.manual_seed_all(1)


class MyUnidirectionalRNN(nn.Module):

  def __init__(self, num_feats, rnn_size=128):
    super().__init__()
    self.rnn_size = rnn_size

    # Definition of the RNN parameters with the use of Linear layers:

    # Define the input activation matrix W
    self.W = nn.Linear(num_feats, rnn_size, bias=False)

    # TODO: Define the hidden activation matrix U
    self.U = nn.Linear(rnn_size, rnn_size, bias=False)

    # Define the bias
    self.b = nn.Parameter(torch.Tensor([0.1]))

  def forward(self, x, state=None):
    # Assuming x is of shape [batch_size, seq_len, num_feats]
    xs = torch.chunk(x, x.shape[1], dim=1)
    print('{}',xs[0].shape)
    print('{}',xs[1].shape)
    hts = []
    if state is None:
      state = self.init_state(x.shape[0])
    ht = state
    for xt in xs:
      # turn x[t] into shape [batch_size, num_feats] to be projected
      xt = xt.squeeze(1)
      ct = self.W(xt)
      ct = ct + self.U(ht)
      ht = ct + self.b
      # give the temporal dimension back to h[t] to be cated
      hts.append(ht.unsqueeze(1))
    hts = torch.cat(hts, dim=1)
    return hts

  def init_state(self, batch_size):
    return torch.zeros(batch_size, self.rnn_size)

# To correctly assess the answer, we build an example RNN with 10 inputs and 32 neurons
rnn = MyUnidirectionalRNN(1, 1)
# Then we will forward 10 random sequences, each of length 15
rnn.eval()
xt = torch.Tensor([[[12],[8]]])
print(xt.shape)
# The returned tensor will be h[t]
ht = rnn(xt)
assert ht.shape[0] == 5 and ht.shape[1] == 15 and ht.shape[2] == 32, \
'Something went wrong within the RNN :('
print('Success! Output shape: {} sequences, each of length {}, each '\
      'token with {} dims'.format(ht.shape[0], ht.shape[1], ht.shape[2]))