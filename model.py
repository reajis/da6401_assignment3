"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
"""
import os
import gdown

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Scaled Dot-Product Attention
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    mask:
        BoolTensor broadcastable to (..., seq_q, seq_k)
        True means masked out.
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # Safety: if an entire row is masked, softmax can become NaN.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════
# Mask Helpers
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    src: [batch, src_len]

    returns: [batch, 1, 1, src_len]
    True means masked out.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    tgt: [batch, tgt_len]

    returns: [batch, 1, tgt_len, tgt_len]
    True means masked out.
    """
    batch_size, tgt_len = tgt.shape
    device = tgt.device

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    # Broadcast OR gives [batch, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
# Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention without using torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Useful for attention visualization in report.
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads:
        # [batch, seq_len, d_model] -> [batch, heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        self.attention_weights = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads:
        # [batch, heads, seq_len, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
# Positional Encoding
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Important: buffer, not trainable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# Position-wise Feed Forward
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# Encoder Layer
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Encoder layer:
    Self-attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# ══════════════════════════════════════════════════════════════════════
# Decoder Layer
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Decoder layer:
    Masked self-attention -> Add & Norm
    Cross-attention -> Add & Norm
    FFN -> Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


# ══════════════════════════════════════════════════════════════════════
# Encoder and Decoder Stacks
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Stack of N encoder layers with final LayerNorm.
    """

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """
    Stack of N decoder layers with final LayerNorm.
    """

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
# Full Transformer
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer.
    """

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        checkpoint_path: str = "checkpoint.pt",
    ) -> None:
        super().__init__()

        checkpoint = None

        # If called as Transformer(), download/load checkpoint first.
        if src_vocab_size is None or tgt_vocab_size is None:
            if checkpoint_path is None:
                raise ValueError(
                    "src_vocab_size and tgt_vocab_size are required unless checkpoint_path is provided."
                )

            # DOWNLOAD checkpoint.pt if it is not already present
            if not os.path.exists(checkpoint_path):
                gdown.download(
                    id="10hUsOjjzqWutrHBhXjIrl4S5wu6IkLsf",
                    output=checkpoint_path,
                    quiet=False,
                )

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            config = checkpoint["model_config"]

            src_vocab_size = config["src_vocab_size"]
            tgt_vocab_size = config["tgt_vocab_size"]
            d_model = config["d_model"]
            N = config["N"]
            num_heads = config["num_heads"]
            d_ff = config["d_ff"]
            dropout = config["dropout"]

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_itos = None
        self.tgt_itos = None

        # If model was created with explicit vocab sizes but checkpoint exists, optionally load it.
        if checkpoint is None and checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if checkpoint is not None:
            self.load_state_dict(checkpoint["model_state_dict"])

            self.src_vocab = checkpoint.get("src_vocab", None)
            self.tgt_vocab = checkpoint.get("tgt_vocab", None)
            self.src_itos = checkpoint.get("src_itos", None)
            self.tgt_itos = checkpoint.get("tgt_itos", None)
    def _reset_parameters(self) -> None:
        """
        Xavier initialization, commonly used for Transformer weights.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ── AUTOGRADER HOOKS ──

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)

        memory = self.encoder(src_emb, src_mask)

        return memory

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        decoder_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)

        logits = self.generator(decoder_out)

        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)

        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translate one German sentence to English using greedy decoding.
        Called by autograder as model.infer(sentence).
        """
        import spacy

        if self.src_vocab is None or self.tgt_itos is None:
            raise ValueError(
                "Vocab not found in checkpoint. Save src_vocab and tgt_itos inside checkpoint."
            )

        device = next(self.parameters()).device

        try:
            spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            spacy_de = spacy.blank("de")
            
        unk_idx = 0
        pad_idx = 1
        sos_idx = 2
        eos_idx = 3

        tokens = [tok.text.lower() for tok in spacy_de.tokenizer(src_sentence)]

        src_indices = [sos_idx]
        src_indices += [self.src_vocab.get(tok, unk_idx) for tok in tokens]
        src_indices += [eos_idx]

        src = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

        self.eval()

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(sos_idx)

            for _ in range(100 - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=pad_idx).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)

                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                ys = torch.cat(
                    [ys, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                    dim=1,
                )

                if next_token == eos_idx:
                    break

        output_tokens = []

        for idx in ys.squeeze(0).tolist():
            if idx in [pad_idx, sos_idx]:
                continue
            if idx == eos_idx:
                break

            if isinstance(self.tgt_itos, dict):
                token = self.tgt_itos.get(idx, "<unk>")
            else:
                token = self.tgt_itos[idx]

            output_tokens.append(token)

        return " ".join(output_tokens)