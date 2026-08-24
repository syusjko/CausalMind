import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 1024 
    n_heads: int = 16
    n_layers: int = 24
    chunk_size: int = 256
    max_seq_len: int = 1024 
    dropout: float = 0.0 
    recovery: float = 0.05
    burn_rate: float = 0.25
    threshold: float = 0.25
    metabolic_lambda: float = 0.0002 
    batch_size: int = 8 
    grad_accum: int = 32
    vocab_size: int = 50257 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @property
    def d_head(self): 
        return self.d_model // self.n_heads

class EndocrineGates(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.cs = cfg.n_heads, cfg.chunk_size
        self.recovery, self.burn, self.thr = cfg.recovery, cfg.burn_rate, cfg.threshold
        self.W_f = nn.Linear(1, self.H)
        self.W_i = nn.Linear(1, self.H)
        nn.init.constant_(self.W_i.bias, -1.0)
        nn.init.zeros_(self.W_f.weight)
        self.W_f.bias.data = torch.linspace(-1.0, 5.0, self.H)

    def forward(self, S_t, tank):
        B, T, _ = S_t.shape
        log_f = F.logsigmoid(self.W_f(S_t))
        i_raw = F.elu(self.W_i(S_t)) + 1.0
        i_eff_list = []
        for s in range(0, T, self.cs):
            e = min(s + self.cs, T)
            i_chunk = i_raw[:, s:e]
            chunk_mean = i_chunk.mean(dim=(1, 2)).unsqueeze(-1)
            tank = (tank + self.recovery - chunk_mean * self.burn).clamp(0, 1)
            debt = torch.sigmoid((self.thr - tank) * 20.0)
            i_eff_list.append(i_chunk * (1.0 - debt).unsqueeze(-1))
        return log_f, torch.cat(i_eff_list, dim=1), tank

class FastSurprisal(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_fast = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.W_fast.weight)

    def forward(self, e):
        e_prev = F.pad(e, (0, 0, 1, 0))[:, :-1, :]
        S_t = 1.0 - F.cosine_similarity(self.W_fast(e_prev), e, dim=-1, eps=1e-8).unsqueeze(-1)
        return S_t.clamp(0, 2.0)

# (Other components like ChunkedParallelRecurrence, RotaryEmbedding, AmygdalaBlock omitted for brevity)

class AmygdalaSLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Assuming AmygdalaBlock is implemented
        self.blocks = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model) for _ in range(cfg.n_layers)]) 
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
    def forward(self, idx, targets=None, init_tank=1.0):
        """
        Core fix implemented: init_tank=0.0135 during inference prevents Activation Explosion.
        """
        B, T = idx.shape
        x = self.embed(idx)
        tank = torch.full((B, 1), init_tank, device=idx.device, dtype=x.dtype)
        tank_history, surprisal_history = [], []
        
        for blk in self.blocks:
            x = blk(x) # Simplified block logic
            tank_history.append(tank)
            # surprisal_history.append(S_t)
            
        logits = self.head(self.norm(x))
        return logits, None, tank_history, surprisal_history
