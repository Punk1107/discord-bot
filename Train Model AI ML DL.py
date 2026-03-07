import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# Global Config
VOCAB_SIZE = 32000
SEQ_LEN = 256

DIM = 768
HEADS = 12
LAYERS = 12
DFF_DIM = 3072

BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0

LORA_RANK = 8
LORA_ALPHA = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# DISTRIBUTED SAFE INIT
def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

# DATA SET (REPLACE WITH REAL TOKENS)
class TokenDataset(Dataset):
    def __init__(self, size=200_000):
        self.data = torch.randint(0, VOCAB_SIZE, (size, SEQ_LEN), dtype=torch.long)
    
    def __init__(self):
        return self.data.size(0)
    def __init__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

# LORA LINEAR (FREEZES BASE WEIGHT)
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        
        self.A = nn.Parameter(torch.randn(LORA_RANK, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, LORA_RANK))
        
        self.scale = LORA_ALPHA / LORA_RANK
        
    def forward(self, x):
        base = x @ self.weight.T
        lora = (x @ self.A.T) @ self.B.T
        return base + lora * self.scale
    
# TRANSFORMER BLOCK (PRE-NORM)
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(DIM)
        self.ln2 = nn.LayerNorm(DIM)
        
        self.qkv = LoRALinear(DIM, DIM * 3)
        self.out = LoRALinear(DIM, DIM)
        
        self.ff = nn.Sequential(
            LoRALinear(DIM, FF_DIM),
            nn.GELU(),
            LoRALinear(FF_DIM, DIM),
        )
        
    def forward(self, x):
        # Attention
        h = self.ln1(x)
        qkv = self.qkv(h).chunk(3, dim=-1)
        
        B, T, _ = h.shape
        q, k, v = [
            t.view(B, T, HEADS, DIM // HEADS).transpose(1, 2)
            for t in qkv
        ]
        
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        
        attn = attn.transpose(1, 2).reshape(B, T, DIM)
        x = x + self.out(attn)
        
        # FeedForward
        x = x + self.ff(self.ln2(x))
        return x
    
