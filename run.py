import os
import torch
import requests
import torch.nn as nn
from torch.nn import functional as F

n_head = 6
n_layer = 6
n_embd = 384
block_size = 256
lr=4e-4
steps = 3500
batch_size=64
val_steps = 200
val_interval=500
attn_dropout=0.1
mlp_dropout = 0.2
label_smoothing=0.02

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, mlp_dropout, attn_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(mlp_dropout),
        )

        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.mlp[2].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, mask=None):
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)[0]
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class CharLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, mlp_dropout, attn_dropout):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        
        self.drop = nn.Dropout(mlp_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, mlp_dropout, attn_dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        
        self.block_size = block_size
        
    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward sequence length > block_size"
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(tok_emb + pos_emb)
        
        x = x.transpose(0, 1)
        
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        mask = mask.to(idx.device)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

def load_shakespeare():
    filepath = 'shakespeare.txt'
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    if not os.path.exists(filepath):
        print("Downloading Shakespeare dataset...", flush=True)
        response = requests.get(url)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!", flush=True)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset length: {len(text)} characters", flush=True)
    print(f"First 100 characters:\n{text[:100]}", flush=True)
    return text

def print_parameter_counts(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            print(f"{name}: {param_count:,}")
            total_params += param_count
    print(f"\nTotal trainable parameters: {total_params:,}")

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_batch(train_data, val_data, batch_size, block_size, split, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train_model(model, train_data, val_data, steps, batch_size, lr, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    model = model.to(device)
    print_parameter_counts(model)
    
    best_val_loss = float('inf')
    
    model.train()
    total_loss = 0
    
    for step in range(steps):
        x, y = get_batch(train_data, val_data, batch_size, block_size, 'train', device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if step > 0 and step % 100 == 0:
            avg_loss = total_loss / 100
            current_lr = scheduler.get_last_lr()[0]
            print(f'Step {step}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}', flush=True)
            total_loss = 0
            
        if step > 0 and step % val_interval == 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = get_batch(train_data, val_data, batch_size, block_size, 'val', device)
                    logits = model(x)
                    val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            
            avg_val_loss = val_loss / val_steps
            print(f'Step {step}: Val Loss = {avg_val_loss:.4f}', flush=True)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f'New best val loss: {best_val_loss:.4f}', flush=True)
            elif avg_val_loss > best_val_loss * 1.1:
                print('Validation loss increasing significantly, consider early stopping', flush=True)
            
            model.train()

def generate(model, context, max_new_tokens, block_size, device, stoi, itos):
    model.eval()
    
    if isinstance(context, str):
        context = torch.tensor([[stoi[c] for c in context]], dtype=torch.long)
    
    context = context.to(device)
    
    for _ in range(max_new_tokens):
        context_cond = context[:, -block_size:] if context.size(1) > block_size else context
        
        with torch.no_grad():
            logits = model(context_cond)
        
        logits = logits[:, -1, :]
        
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, next_token), dim=1)
    
    generated = ''.join([itos[i.item()] for i in context[0]])
    return generated

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}", flush=True)
    
    text = load_shakespeare()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    
    train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    
    print(f"Vocabulary size: {vocab_size}", flush=True)
    print(f"Train data length: {len(train_data)}", flush=True)
    print(f"Val data length: {len(val_data)}", flush=True)
    
    model = CharLM(vocab_size, n_embd, n_head, n_layer, block_size, mlp_dropout, attn_dropout)
    train_model(model, train_data, val_data, steps, batch_size, lr, device)

    context = torch.zeros((1, 1), dtype=torch.long)  # or start with some text
    generated_text = generate(model, context, max_new_tokens=500, block_size=block_size, 
                            device=device, stoi=stoi, itos=itos)
    print("\nGenerated text:\n", generated_text)
