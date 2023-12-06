import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import matplotlib.pyplot as plt
from timeit import default_timer as timer

torch.manual_seed(1)

device = 'cpu'
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  device = torch.device("mps")
  torch.cuda.manual_seed_all(1)

print(device)

# Let's prepare some synthetic data

def prepare_sequence(seq, char2idx, onehot=True):
    # convert sequence of words to indices
    idxs = [char2idx[c] for c in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    if onehot:
      # conver to onehot (if input to network)
      ohs = F.one_hot(idxs, len(char2idx)).float()
      return ohs
    else:
      return idxs

with open('episode1_english.txt', 'r') as txt_f:
  training_data = [l.rstrip() for l in txt_f if l.rstrip() != '']

# merge the training data into one big text line
training_data = '$'.join(training_data)

# Assign a unique ID to each different character found in the training set
char2idx = {}
for c in training_data:
    if c not in char2idx:
        char2idx[c] = len(char2idx)
idx2char = dict((v, k) for k, v in char2idx.items())
VOCAB_SIZE = len(char2idx)
RNN_SIZE = 1024
MLP_SIZE = 2048
SEQ_LEN = 50
print('Number of found vocabulary tokens: ', VOCAB_SIZE)


class CharLSTM(nn.Module):

  def __init__(self, vocab_size, rnn_size, mlp_size):
    super().__init__()
    self.rnn_size = rnn_size

    # TODO: Define the LSTM
    self.lstm = nn.LSTM(vocab_size, rnn_size, batch_first=True)

    self.dout = nn.Dropout(0.4)

    # TODO: Create an MLP with a hidden layer of mlp_size neurons that maps
    # from the RNN hidden state space to the output space of vocab_size
    self.mlp = nn.Sequential(
      # Linear layer
      nn.Linear(rnn_size, mlp_size),
      # Activation function
      nn.ReLU(),
      # Dropout (0.4)
      nn.Dropout(0.4),
      # Linear layer
      nn.Linear(mlp_size, vocab_size),
      # Activation function
      nn.LogSoftmax(dim=1),
    )

  def forward(self, sentence, state=None):
    bsz, slen, vocab = sentence.shape
    ht, state = self.lstm(sentence, state)
    ht = self.dout(ht)
    h = ht.contiguous().view(-1, self.rnn_size)
    logprob = self.mlp(h)
    return logprob, state

# Let's build an example model and see what the scores are before training
model = CharLSTM(VOCAB_SIZE, RNN_SIZE, MLP_SIZE)

# This should output crap as it is not trained, so a fixed random tag for everything

def gen_text(model, seed, char2idx, num_chars=150):
  model.eval()
  # Here we don't need to train, so the code is wrapped in torch.no_grad()
  with torch.no_grad():
    inputs = prepare_sequence(seed, char2idx)
    # fill the RNN memory with the seed sentence
    seed_pred, state = model(inputs.unsqueeze(0))
    # now begin looping with feedback char by char from the last prediction
    preds = seed
    curr_pred = torch.argmax(seed_pred[-1, :])
    curr_pred = idx2char[curr_pred.item()]
    preds += curr_pred
    for _ in range(num_chars):

      # TODO: Get the next char prediction from the model given the current prediction and current state
      # inputs = prepare_sequence(preds, char2idx)
      # inputs = prepare_sequence(preds.replace('\n', '$'), char2idx)
      inputs = prepare_sequence(curr_pred, char2idx)
      curr_pred, state = model(inputs.unsqueeze(0), state)

      curr_pred = torch.argmax(curr_pred[-1, :])
      curr_pred = idx2char[curr_pred.item()]
      if curr_pred == '$':
        # special token to add newline char
        preds += '\n'
        # preds += ' '
      else:
        preds += curr_pred
    return preds

#print(gen_text(model, 'Monica was ', char2idx))

BATCH_SIZE = 64
T = len(training_data)
CHUNK_SIZE = T // BATCH_SIZE
# let's first chunk the huge train sequence into BATCH_SIZE sub-sequences
trainset = [training_data[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, T - CHUNK_SIZE, CHUNK_SIZE),
                                    range(CHUNK_SIZE, T, CHUNK_SIZE))]
print('Original training string len: ', T)
print('Sub-sequences len: ', CHUNK_SIZE)

# The way training works is the following:
# at each batch sampling from the trainset, we pick a portion of sequences
# continuous with a sliding window in time. Hence, each of the BATCH_SIZE sub-sequences
# in batch b[i] will continue in batch b[i + 1] in the same position of the batch dimension.
# This is called stateful sampling, where we train with consecutive windows of sequences
# We broke the long string into BATCH_SIZE subsequence, so we introduced BATCH_SIZE - 1
# discontinuities... YES. But we can assume that each sub-sequence is continuous in a long
# enough chunk so that those discontinuities are negligible.

# Let's now build a model to train with its optimizer and loss
model = CharLSTM(VOCAB_SIZE, RNN_SIZE, MLP_SIZE)
model.to(device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 2000
tr_loss = []
state = None
timer_beg = timer()
for epoch in range(NUM_EPOCHS):
  model.train()
  # let's slide over our dataset
  for beg_t, end_t in zip(range(0, CHUNK_SIZE - SEQ_LEN - 1, SEQ_LEN + 1),
                          range(SEQ_LEN + 1, CHUNK_SIZE, SEQ_LEN + 1)):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    optimizer.zero_grad()

    dataX = []
    dataY = []
    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of one-hot sequences.
    for sent in trainset:
      # chunk the sentence
      chunk = sent[beg_t:end_t]

      # TODO: get X and Y with a shift of 1
      X = chunk[:-1]  # remove last char
      Y = chunk[1:]  # remove first char

      # convert each sequence to one-hots and labels respectively
      X = prepare_sequence(X, char2idx)
      Y = prepare_sequence(Y, char2idx, onehot=False)
      dataX.append(X.unsqueeze(0))  # create batch-dim
      dataY.append(Y.unsqueeze(0))  # create batch-dim
    dataX = torch.cat(dataX, dim=0).to(device)
    dataY = torch.cat(dataY, dim=0).to(device)

    # Step 3. Run our forward pass.
    # Forward through model and carry the previous state forward in time (statefulness)
    y_, state = model(dataX, state)
    # detach the previous state graph to not backprop gradients further than the BPTT span
    state = (state[0].detach(),  # detach c[t]
             state[1].detach())  # detach h[t]

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(y_, dataY.view(-1))
    loss.backward()
    optimizer.step()
    tr_loss.append(loss.item())
  timer_end = timer()
  if (epoch + 1) % 50 == 0:
    # Generate a seed sentence to play around
    model.to('cpu')
    print('-' * 30)
    print(gen_text(model, 'They ', char2idx))
    print('-' * 30)
    model.to(device)
    print('Finished epoch {} in {:.1f} s: loss: {:.6f}'.format(epoch + 1,
                                                               timer_end - timer_beg,
                                                               np.mean(tr_loss[-10:])))
  timer_beg = timer()

plt.plot(tr_loss)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')

print(gen_text(model.to('cpu'), 'Professor ', char2idx, 1000))