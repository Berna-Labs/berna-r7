"""
Berna R4 - Prototype Model Architecture
Decoder-only Transformer, HuggingFace-compatible
Trained from scratch - no fine-tuning of other models.
"""
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = nn.Module
    PretrainedConfig = object
    CausalLMOutputWithPast = dict


class BernaConfig(PretrainedConfig):
    model_type = "berna"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: BernaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value, seq_len=q_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        # Use SDPA (Flash Attention on Blackwell) - memory efficient
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config: BernaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: BernaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BernaModel(nn.Module):
    def __init__(self, config: BernaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class BernaForCausalLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    config_class = BernaConfig

    def __init__(self, config: BernaConfig):
        super().__init__(config)
        self.model = BernaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if HF_AVAILABLE:
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50, **kwargs):
        """
        Safe generation with NaN/Inf handling.
        Works even with early-training models (unstable logits).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_position_embeddings:]
            outputs = self.forward(idx_cond)
            logits = outputs.logits if HF_AVAILABLE else outputs["logits"]
            logits = logits[:, -1, :]  # (batch, vocab)

            # Sanitize: replace NaN/Inf with 0
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering (safe)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, top_idx = torch.topk(logits, k, dim=-1)
                # Create new logits filled with -inf
                new_logits = torch.full_like(logits, float("-inf"))
                # Scatter top-k values back
                new_logits.scatter_(-1, top_idx, top_vals)
                logits = new_logits

            # Softmax (now safe because at least k values are finite)
            probs = F.softmax(logits, dim=-1)

            # Handle any remaining NaN (fallback to uniform)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = BernaConfig()
    model = BernaForCausalLM(config)
    n_params = count_parameters(model)
    print("=" * 60)
    print("Berna R4 Prototype")
    print("=" * 60)
    print(f"Parameters: {n_params / 1e6:.2f}M")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Hidden: {config.hidden_size}")
    print(f"Heads: {config.num_attention_heads}")
    print(f"Vocab: {config.vocab_size}")
    print(f"Context: {config.max_position_embeddings}")
    print("=" * 60)
