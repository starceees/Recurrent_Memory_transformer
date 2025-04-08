# model/model.py
from __future__ import annotations
import math
from functools import partial
from itertools import zip_longest
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# Import Attend from a third-party package
from recurrent_memory_transformer_pytorch.attend import Attend
from hyper_connections import get_init_and_expand_reduce_stream_functions

Linear = partial(nn.Linear, bias=False)

# Helper functions
def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def divisible_by(numer, denom):
    return (numer % denom) == 0

# Sampling helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def frac_gradient(t, frac=1.):
    if frac == 1.:
        return t
    return t * frac + t.detach() * (1. - frac)

# Rotary Embedding
class RotaryEmbedding(Module):
    def __init__(self, dim, theta=32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# Feedforward network
class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

def FeedForward(dim, mult=4, dropout=0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

# Attention layer
class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.,
        accept_value_residual=False,
        use_flash_attn=False,
        use_custom_causal_attn_mask=False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        dim_inner = dim_head * heads
        self.heads = heads
        self.attend = Attend(
            causal=causal and not use_custom_causal_attn_mask,
            dropout=dropout,
            use_flash=use_flash_attn
        )
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))
        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)
        self.learned_value_residual_mix = None
        if accept_value_residual:
            self.learned_value_residual_mix = nn.Sequential(
                Linear(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

    def forward(
        self,
        x,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        mask=None,
        xl_memories=None,
        value_residual=None
    ):
        assert not (exists(value_residual) ^ exists(self.learned_value_residual_mix))
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        orig_v = v
        if exists(self.learned_value_residual_mix):
            mix = self.learned_value_residual_mix(x)
            v = v.lerp(value_residual, mix)
        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b=x.shape[0]), self.null_kv)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
        next_xl_memories = torch.stack((k, v))
        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim=-2)
            v = torch.cat((vx, v), dim=-2)
            if exists(mask):
                mask = F.pad(mask, (xl_memories.shape[-2], 0), value=True)
        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb
            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)
        out = self.attend(q, k, v, mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), next_xl_memories, orig_v

# Transformer architecture
class RecurrentMemoryTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        causal=True,        
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
        use_flash_attn=False,
        ignore_index=-1,
        abs_pos_emb=True,
        rotary_pos_emb=False,
        use_xl_memories=True,
        xl_mem_len=None,
        enhanced_xl_recurrence=False,
        emb_gradient_frac=0.1,
        memory_not_causal=True,
        add_write_to_next_write_mem=False,
        next_write_mem_stop_grad=True,
        always_have_read_memories=True,
        num_residual_streams=4
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len
        self.emb_gradient_frac = emb_gradient_frac
        assert num_memory_tokens > 0
        self.token_emb = nn.Embedding(num_tokens, dim)
        assert any([abs_pos_emb, rotary_pos_emb])
        self.pos_emb = nn.Embedding(seq_len, dim) if abs_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None
        self.num_memory_tokens = num_memory_tokens
        self.read_memory_emb = nn.Parameter(torch.zeros(num_memory_tokens, dim))
        nn.init.normal_(self.read_memory_emb, std=0.02)
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std=0.02)
        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len
        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence
        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable=(num_residual_streams == 1))
        self.layers = ModuleList([])
        for layer_index in range(depth):
            is_first = layer_index == 0
            self.layers.append(ModuleList([
                init_hyper_conn(dim=dim, branch=Attention(
                    dim=dim,
                    dim_head=dim_head,
                    causal=causal,
                    heads=heads,
                    use_flash_attn=use_flash_attn,
                    accept_value_residual=not is_first,
                    use_custom_causal_attn_mask=memory_not_causal,
                    dropout=attn_dropout
                )),
                init_hyper_conn(dim=dim, branch=FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)),
            ]))
        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)
        self.ignore_index = ignore_index
        self.use_custom_causal_attn_mask = causal and memory_not_causal
        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad
        self.always_have_read_memories = always_have_read_memories

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b=batch)

    def forward(
        self,
        x,
        read_memories=None,
        *,
        mask=None,
        labels=None,
        xl_memories: list[Tensor] | None = None,
        mask_out_read_memories=False
    ):
        has_xl_memories = exists(xl_memories) and len(xl_memories) > 0
        b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)
        assert n <= self.seq_len
        pos = torch.arange(n, device=device)
        x = self.token_emb(x)
        if exists(self.pos_emb):
            x = x + self.pos_emb(pos)
        x = frac_gradient(x, self.emb_gradient_frac)
        write_memories = self.init_memory(b)
        if exists(read_memories) and self.add_write_to_next_write_mem:
            maybe_detach = torch.detach if self.next_write_mem_stop_grad else identity
            write_memories = write_memories + maybe_detach(read_memories)
        if exists(read_memories):
            if read_memories.ndim == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b=b)
            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        elif self.always_have_read_memories:
            read_mem_length = mem_length
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b=b)
        else:
            read_mem_length = 0
            read_memories = x[:, 0:0]
        x, ps = pack([read_memories, x, write_memories], 'b * d')
        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value=True)
        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).tril()
            causal_mask = F.pad(causal_mask, (0, mem_length, read_mem_length, 0), value=False)
            causal_mask = F.pad(causal_mask, (read_mem_length, 0, 0, mem_length), value=True)
            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')
            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask & causal_mask
            else:
                mask = causal_mask
        if read_mem_length > 0 and mask_out_read_memories:
            read_mem_mask = torch.arange(x.shape[-2], device=device) < read_mem_length
            if exists(mask):
                mask = mask & ~read_mem_mask
            else:
                mask = read_mem_mask
        rotary_emb = None
        if exists(self.rotary_pos_emb):
            mem_rel_dist = 10000
            q_pos = pos + mem_rel_dist
            if has_xl_memories:
                xl_mem_length = xl_memories[0].shape[-2]
                q_pos += xl_mem_length
            q_pos = F.pad(q_pos, (read_mem_length, mem_length), value=0)
            q_rotary_emb = self.rotary_pos_emb(q_pos_
