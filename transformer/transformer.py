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

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=5000):
        """
        Args:
            embed_dim (int): Embedding dimensionality
            max_len (int): Maximum length of a sequence to expect
        """
        super(PositionalEncoding, self).__init__()

        # Create matrix of (T x embed_dim) representing the positional encoding
        # for max_len inputs
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, ffn_dim, num_heads, dropout=0.0):
        """
        Args:
            embed_dim (int): Embedding dimensionality (input, output & self-attention)
            ffn_dim (int): Inner dimensionality in the FFN
            num_heads (int): Number of heads of the multi-head attention block
            dropout (float): Dropout probability
        """
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_att=False):
        src_len, batch_size, _ = x.shape
        if mask is None:
            mask = torch.zeros(x.shape[1], x.shape[0]).bool().to(x.device)

        selfattn_mask = mask.unsqueeze(-2)

        # TODO: Self-Attention block
        selfattn_out, selfattn_w = self.self_attn(x, x, x, selfattn_mask)
        selfattn_out = self.dropout(selfattn_out)

        # TODO: Add + normalize block (1)
        x = self.norm1(x + selfattn_out)

        # TODO: FFN block
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)

        # TODO: Add + normalize block (2)
        x = self.norm2(x + ffn_out)

        if return_att:
            return x, selfattn_w
        else:
            return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, embed_dim, ffn_dim, num_heads, vocab_size, dropout=0.0):
        super(TransformerEncoder, self).__init__()

        # Create an embedding table (T x B -> T x B x embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Create the positional encoding with the class defined before
        self.pos_enc = PositionalEncoding(embed_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, return_att=False):
        x = self.embedding(x)
        x = self.pos_enc(x)

        selfattn_ws = []
        for l in self.layers:
            if return_att:
                x, selfattn_w = l(x, mask=mask, return_att=True)
                selfattn_ws.append(selfattn_w)
            else:
                x = l(x, mask=mask, return_att=False)

        if return_att:
            selfattn_ws = torch.stack(selfattn_ws, dim=1)
            return x, selfattn_ws
        else:
            return x

transformer_encoder_cfg = {
    "num_layers": 6,
    "embed_dim": 512,
    "ffn_dim": 2048,
    "num_heads": 8,
    "vocab_size": 8000,
    "dropout": 0.1,
}

transformer_encoder = TransformerEncoder(**transformer_encoder_cfg)

src_batch_example = torch.randint(transformer_encoder_cfg['vocab_size'], (20, 4))

encoder_out, attn_ws = transformer_encoder(src_batch_example, return_att=True)

print(f"Encoder output: {encoder_out.shape}")
print(f"Self-Attention weights: {attn_ws.shape}")

class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, ffn_dim, num_heads, dropout=0.0):
        """
        Args:
            embed_dim (int): Embedding dimensionality (input, output & self-attention)
            ffn_dim (int): Inner dimensionality in the FFN
            num_heads (int): Number of heads of the multi-head attention block
            dropout (float): Dropout probability
        """
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.encdec_attn = MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask=None, memory_mask=None, return_att=False):
        tgt_len, batch_size, _ = x.shape
        src_len, _, _ = memory.shape
        if mask is None:
            mask = torch.zeros(x.shape[1], x.shape[0])
            mask = mask.bool().to(x.device)
        if memory_mask is None:
            memory_mask = torch.zeros(memory.shape[1], memory.shape[0])
            memory_mask = memory_mask.bool().to(memory.device)


        subsequent_mask = torch.triu(torch.ones(batch_size, tgt_len, tgt_len), 1)
        subsequent_mask = subsequent_mask.bool().to(mask.device)
        selfattn_mask = subsequent_mask + mask.unsqueeze(-2)

        attn_mask = memory_mask.unsqueeze(-2)

        # Self-Attention block
        selfattn_out, selfattn_w = self.self_attn(x, x, x, selfattn_mask)
        selfattn_out = self.dropout(selfattn_out)

        # Add + normalize block (1)
        x = self.norm1(x + selfattn_out)

        # Encoder-Decoder Attention block
        attn_out, attn_w = self.encdec_attn(x, memory, memory, attn_mask)
        attn_out = self.dropout(attn_out)

        # TODO: Add + normalize block (2)
        x = self.norm2(x + attn_out)

        # TODO: FFN block
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)

        # TODO: Add + normalize block (3)
        x = self.norm3(x + ffn_out)

        if return_att:
            return x, selfattn_w, attn_w
        else:
            return x

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, embed_dim, ffn_dim, num_heads, vocab_size, dropout=0.0):
        super(TransformerDecoder, self).__init__()

        # Create an embedding table (T x B -> T x B x embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Create the positional encoding with the class defined before
        self.pos_enc = PositionalEncoding(embed_dim)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Add a projection layer (T x B x embed_dim -> T x B x vocab_size)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, memory, mask=None, memory_mask=None, return_att=False):
        x = self.embedding(x)
        x = self.pos_enc(x)

        selfattn_ws = []
        attn_ws = []
        for l in self.layers:
            if return_att:
                x, selfattn_w, attn_w = l(
                    x, memory, mask=mask, memory_mask=memory_mask, return_att=True
                )
                selfattn_ws.append(selfattn_w)
                attn_ws.append(attn_w)
            else:
                x = l(
                    x, memory, mask=mask, memory_mask=memory_mask, return_att=False
                )

        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)

        if return_att:
            selfattn_ws = torch.stack(selfattn_ws, dim=1)
            attn_ws = torch.stack(attn_ws, dim=1)
            return x, selfattn_ws, attn_ws
        else:
            return x

transformer_decoder_cfg = {
    "num_layers": 6,
    "embed_dim": 512,
    "ffn_dim": 2048,
    "num_heads": 8,
    "vocab_size": 8000,
    "dropout": 0.1,
}

transformer_decoder = TransformerDecoder(**transformer_decoder_cfg)

tgt_batch_example = torch.randint(transformer_decoder_cfg['vocab_size'], (15, 4))

decoder_out, selfattn_ws, attn_ws  = transformer_decoder(
    tgt_batch_example,
    memory=encoder_out,
    return_att=True
)

print(f"Decoder output: {decoder_out.shape}")
print(f"Self-Attention weights: {selfattn_ws.shape}")
print(f"Enc-Dec Attention weights: {attn_ws.shape}")


class Transformer(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(**encoder_config)
        self.decoder = TransformerDecoder(**decoder_config)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """ Forward method

        Method used at training time, when the target is known. The target tensor
        passed to the decoder is shifted to the right (starting with BOS
        symbol). Then, the output of the decoder starts directly with the first
        token of the sentence.
        """

        # TODO: Compute the encoder output
        encoder_out = self.encoder(src, mask=src_mask)

        # TODO: Compute the decoder output
        decoder_out = self.decoder(
            x=tgt,
            memory=encoder_out,
            mask=tgt_mask,
            memory_mask=src_mask
        )

        return decoder_out

    def generate(self, src, src_mask=None, bos_idx=0, max_len=50):
        """ Generate method

        Method used at inference time, when the target is unknown. It
        iteratively passes to the decoder the sequence generated so far
        and appends the new token to the input again. It uses a Greedy
        decoding (argmax).
        """

        # TODO: Compute the encoder output
        encoder_out = self.encoder(src, mask=src_mask)

        output = torch.LongTensor([bos_idx])\
                    .expand(1, encoder_out.size(1)).to(src.device)
        for i in range(max_len):
            # TODO: Get the new token
            new_token = self.decoder(
                x=output[-1, :].unsqueeze(0),
                memory=encoder_out,
                memory_mask=src_mask
            )[-1].argmax(-1)

            output = torch.cat([output, new_token.unsqueeze(0)], dim=0)

        return output

transformer = Transformer(transformer_encoder_cfg, transformer_decoder_cfg)

transformer(src_batch_example, tgt_batch_example).shape

from seq2seq_numbers_dataset import generate_dataset_pytorch, Seq2SeqNumbersCollater

numbers_dataset = generate_dataset_pytorch()

# Downsample the dataset to reduce training time (remove for better performance)
numbers_dataset['train'].src_sents = numbers_dataset['train'].src_sents[:25000]
numbers_dataset['train'].tgt_sents = numbers_dataset['train'].tgt_sents[:25000]

collater = Seq2SeqNumbersCollater(
    numbers_dataset['train'].src_dict,
    numbers_dataset['train'].tgt_dict,
)

lr = 5e-4
batch_size = 32
log_interval = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

numbers_loader_train = DataLoader(
    numbers_dataset['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collater,
)

src_dict = numbers_dataset['train'].src_dict
tgt_dict = numbers_dataset['train'].tgt_dict

transformer_encoder_cfg = {
    "num_layers": 3,
    "embed_dim": 256,
    "ffn_dim": 1024,
    "num_heads": 4,
    "vocab_size": len(src_dict),
    "dropout": 0.1,
}
transformer_decoder_cfg = {
    "num_layers": 3,
    "embed_dim": 256,
    "ffn_dim": 1024,
    "num_heads": 4,
    "vocab_size": len(tgt_dict),
    "dropout": 0.1,
}
model = Transformer(transformer_encoder_cfg, transformer_decoder_cfg)
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = F.nll_loss

print("Training model...")

loss_avg = 0
for i, (src, tgt) in enumerate(numbers_loader_train):
    src = {k: v.to(device) for k, v in src.items()}
    tgt = {k: v.to(device) for k, v in tgt.items()}

    optimizer.zero_grad()

    output = model(
        src['ids'],
        tgt['ids'][:-1],
        src['padding_mask'],
        tgt['padding_mask'][:, :-1],
    )

    loss = criterion(
        output.reshape(-1, output.size(-1)),
        tgt['ids'][1:].flatten()
    )
    loss.backward()
    optimizer.step()

    loss_avg += loss.item()
    if (i+1) % log_interval == 0:
        loss_avg /= log_interval
        print(f"{i+1}/{len(numbers_loader_train)}\tLoss: {loss_avg}")

batch_size_test = 128
log_interval_test = 50

numbers_loader_test = DataLoader(
    numbers_dataset['test'],
    batch_size=batch_size_test,
    shuffle=False,
    collate_fn=collater,
)

model.eval()

print("\nTesting model...")

n_correct = 0
n_total = 0
for i, (src, tgt) in enumerate(numbers_loader_test):
    src = {k: v.to(device) for k, v in src.items()}
    tgt = {k: v.to(device) for k, v in tgt.items()}

    output = model.generate(
        src['ids'],
        src_mask=src['padding_mask'],
        bos_idx=numbers_dataset['test'].tgt_dict.bos_idx(),
    )
    output = output[:tgt['ids'].size(0)]

    n_correct += torch.eq(tgt['ids'], output).sum()
    n_total += tgt['ids'].numel()
    if (i+1) % log_interval_test == 0:
        print(f"{i+1}/{len(numbers_loader_test)}")

print(f"Test Accuracy: {100 * n_correct / n_total}%")