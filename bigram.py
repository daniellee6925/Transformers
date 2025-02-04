"""
This is a python script for the bigram language model
"""

# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# hpyerparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # creates embedding with vocab size and embedding dim of n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idex and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
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
            # get the predictions
            logits, loss = self(idx)  # loss will be ignored (uses forward function)
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
