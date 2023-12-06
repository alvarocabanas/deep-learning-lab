"""
curl -o cat-eng.zip http://www.manythings.org/anki/cat-eng.zip
unzip cat-eng.zip && rm cat-eng.zip
mkdir data
mv "cat.txt" "data/eng-cat.txt"
rm "_about.txt"
"""

from __future__ import unicode_literals, print_function, division

import re
import math
import random
import string
import unicodedata
from io import open

import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

#To ensure reproducibility of the experiments, we can set the seed to a fixed number.
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)
_ = torch.mps.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#  device = torch.device("mps")

print(device)

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Remove extra info
    new_lines = []
    for line in lines:
        new_lines.append(line[0:(line.find('CC-BY')-1)])


    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in new_lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 20

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'cat', False)
print(random.choice(pairs))


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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # TODO: Define the Embedding matrix (use nn.Embedding)
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        # Here we define the attention we use
        self.attn = AdditiveAttention(self.hidden_size, self.hidden_size, self.hidden_size // 2)
        #self.attn = MultiplicativeAttention(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        x, hidden = self.gru(embedded, hidden)
        context, attn_weights = self.attn(query=x, key=encoder_outputs, value=encoder_outputs)

        # TODO: compute concatenation of the decoder hidden states and the context vectors
        x_w_context = torch.cat((x, context), dim=-1)

        x_w_context = self.attn_combine(x_w_context)
        output = F.log_softmax(self.out(x_w_context), dim=-1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # First hidden state used by the decoder is the last encoder hidden state
    decoder_hidden = encoder_hidden

    # We feed the decoder with the whole target sentence (teacher forcing),
    # first we append SOS_token

    # With Teacher Forcing, each next token is generated taking in consideration
    # as previous token the ground truth (target) token.


    decoder_input = torch.cat([
        torch.tensor([[SOS_token]], device=device),
        target_tensor[:, :-1]
    ], dim=1)


    decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

    loss = criterion(
        decoder_output.view(-1, decoder_output.size(-1)),
        target_tensor.view(-1)
    )
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.numel()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Random sample of n_iters pairs
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 50000, print_every=3000)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions.append(decoder_attention.squeeze(0).data)
            topv, topi = decoder_output.data.topk(1)
            if topi.squeeze().item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.squeeze().item()])
            decoder_input = topi.detach().squeeze(1)

        return decoded_words, torch.cat(decoder_attentions)

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('Source:', pair[0])
        print('Reference:', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words[:-1])
        print('Model:', output_sentence)
        print('')


evaluateRandomly(encoder1, attn_decoder1)

def showAttention(input_sentence, output_words, attentions):
    import pandas as pd
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #cax = ax.matshow(attentions.numpy(), cmap='Blues')
    #fig.colorbar(cax)
    sns.set(rc={'figure.figsize':(12, 8)})
    input_sentence_list = input_sentence.split(' ') + ['<EOS>']

    df = pd.DataFrame(attentions, columns = input_sentence_list, index = output_words)
    ax = sns.heatmap(
        df.astype(float),
        linewidth=0.5,
        cmap="Blues",
        square=True)


    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions.detach().cpu())


evaluateAndShowAttention("your son is a genius .")

evaluateAndShowAttention("i can t remember which is my racket .")

evaluateAndShowAttention("please circle the right answer .")

evaluateAndShowAttention("i d like to reserve a table for two .")