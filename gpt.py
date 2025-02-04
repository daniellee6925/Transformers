"""
This is a python script for the bigram language model
"""

# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hpyerparameters
batch_size = 64  # how many independent sequences will be processed in parallel
block_size = 256  # maximum context length (chars)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------------------------

torch.manual_seed(1337)

# open file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from chars to ints
stoi = {ch: i for i, ch in enumerate(chars)}  # string to ints
itos = {i: ch for i, ch in enumerate(chars)}  # ints to strings


def encode(s):
    return [stoi[c] for c in s]  # take a string and output a list of integers


def decode(l):
    return "".join(itos[i] for i in l)  # take a list of ints and output a string


# Train and Test Splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and target y
    data = train_data if split == "train" else val_data
    # generate batch size number of random offsets
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    # x becomes a row in a batch_size x block_szie tensor
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# tells framework to disable gradient computation -> during model evaluation or inference
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # set the model to eval mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # get average loss for both splits
    model.train()  # set the model to train mode
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()  # initialize attributes and behaviors of parent class
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer: not updated during training via backpropagation (it doesn’t have gradients).
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention score "affinities"
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) #(B, C, T) -> (B, T, T)
        # masking: tokens in the future cannot communicate
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # normalize
        wei = self.dropout(wei)
        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate output to channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """simple linear layer followed by non-linearity (RELU)"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # projects embedding into higher dim
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # compresses back into original
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication(multihead attention) followed by computation(FeedForward)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections
        # output = Activation(layer(X) + X)
        """
        Input -> [Self-Attention] -> + (Residual Connection) -> [LayerNorm] ->
        -> [Feedforward Network] -> + (Residual Connection) -> [LayerNorm] -> Output
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # creates embedding with vocab size and embedding dim of n_embd
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idex and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embeddings_table(idx)  # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T, C
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # B: batch_size (number of samples processed togeter)
            # T: Time steps (number of tokens)
            # C: Number of Channels/features (embedding size)
            # loss: negative log likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idex is (B, T) arry of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions,  loss will be ignored (uses forward function)
            logits, loss = self(idx_cond)
            # focus only on the last time step, becomes (B, C) last element in the time dimension -> last character
            logits = logits[:, -1, :]
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution, (B, 1) single prediction for what comes next
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# create model
model = BigramLanguageModel()
bigram_model = model.to(device)

# create a pytorch optimizer (Adam)
optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)

# run the training loop
for iter in range(max_iters):
    # every once is a whle, evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = bigram_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(bigram_model.generate(context, max_new_tokens=500)[0].tolist()))
