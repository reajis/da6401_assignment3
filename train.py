"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import math
import os
import argparse
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset, collate_fn


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in the range [0, 1).")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_idx] = 0.0

            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            pad_mask = target.eq(self.pad_idx)
            true_dist.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        loss = -(true_dist * log_probs).sum(dim=1)

        non_pad_mask = target.ne(self.pad_idx)
        if non_pad_mask.sum() == 0:
            return loss.sum() * 0.0

        return loss.masked_select(non_pad_mask).mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    pad_idx = getattr(loss_fn, "pad_idx", 1)

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx).to(device)

        if is_train:
            if optimizer is None:
                raise ValueError("optimizer must not be None when is_train=True")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)

            loss = loss_fn(logits_flat, tgt_output_flat)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        num_tokens = tgt_output.ne(pad_idx).sum().item()
        total_loss += loss.item() * max(num_tokens, 1)
        total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(start_symbol)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)

            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            next_token_tensor = torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_token)

            ys = torch.cat([ys, next_token_tensor], dim=1)

            if next_token == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)

    if hasattr(vocab, "itos"):
        return vocab.itos[idx]

    if isinstance(vocab, dict):
        if idx in vocab:
            return vocab[idx]

        inverse_vocab = {v: k for k, v in vocab.items()}
        return inverse_vocab.get(idx, "<unk>")

    if isinstance(vocab, list):
        return vocab[idx]

    raise TypeError("tgt_vocab must support lookup_token, .itos, dict, or list access.")


def _tokens_from_indices(indices, vocab, special_tokens=None):
    if special_tokens is None:
        special_tokens = {"<pad>", "<sos>", "<eos>"}

    tokens = []
    for idx in indices:
        token = _lookup_token(vocab, int(idx))

        if token == "<eos>":
            break

        if token not in special_tokens:
            tokens.append(token)

    return tokens


def _sentence_from_indices(indices, vocab):
    return " ".join(_tokens_from_indices(indices, vocab))


def _get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(predictions, references, max_n=4):
    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]

    pred_len = 0
    ref_len = 0

    for pred_tokens, ref_tokens in zip(predictions, references):
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            clipped_counts[n - 1] += sum(
                min(count, ref_ngrams.get(ngram, 0))
                for ngram, count in pred_ngrams.items()
            )
            total_counts[n - 1] += max(len(pred_tokens) - n + 1, 0)

    if pred_len == 0:
        return 0.0

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    smooth_precisions = [
        p if p > 0.0 else 1.0 / (2.0 * max(total_counts[i], 1))
        for i, p in enumerate(precisions)
    ]

    log_precision_sum = sum(math.log(p) for p in smooth_precisions) / max_n

    if pred_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / pred_len))

    return 100.0 * brevity_penalty * math.exp(log_precision_sum)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()

    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i = src[i:i + 1]
                tgt_i = tgt[i]

                src_mask = make_src_mask(src_i, pad_idx=pad_idx).to(device)

                decoded = greedy_decode(
                    model=model,
                    src=src_i,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                pred_tokens = _tokens_from_indices(decoded.squeeze(0).tolist(), tgt_vocab)
                ref_tokens = _tokens_from_indices(tgt_i.tolist(), tgt_vocab)

                predictions.append(pred_tokens)
                references.append(ref_tokens)

    bleu_score = _corpus_bleu(predictions, references)
    return float(bleu_score)


# ══════════════════════════════════════════════════════════════════════
#   W&B REPORT HELPERS
# ══════════════════════════════════════════════════════════════════════

def log_sample_translations(
    model: Transformer,
    dataloader: DataLoader,
    tgt_vocab,
    device: str,
    wandb,
    max_examples: int = 10,
    max_len: int = 100,
) -> None:
    """
    Log sample reference/prediction translations to W&B.
    """
    if wandb is None:
        return

    model.eval()

    rows = []
    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i = src[i:i + 1]
                tgt_i = tgt[i]

                src_mask = make_src_mask(src_i, pad_idx=pad_idx).to(device)

                decoded = greedy_decode(
                    model=model,
                    src=src_i,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                reference = _sentence_from_indices(tgt_i.tolist(), tgt_vocab)
                prediction = _sentence_from_indices(decoded.squeeze(0).tolist(), tgt_vocab)

                rows.append([reference, prediction])

                if len(rows) >= max_examples:
                    table = wandb.Table(
                        columns=["Reference English", "Predicted English"],
                        data=rows,
                    )
                    wandb.log({"sample_translations": table})
                    return


def log_attention_heatmap(
    model: Transformer,
    dataloader: DataLoader,
    tgt_vocab,
    device: str,
    wandb,
    max_len: int = 100,
) -> None:
    """
    Log one decoder cross-attention heatmap to W&B.

    This uses the final decoder layer's cross-attention weights.
    """
    if wandb is None:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()

    pad_idx = 1
    sos_idx = 2
    eos_idx = 3

    with torch.no_grad():
        src, tgt = next(iter(dataloader))
        src = src[:1].to(device)
        tgt = tgt[:1].to(device)

        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

        decoded = greedy_decode(
            model=model,
            src=src,
            src_mask=src_mask,
            max_len=max_len,
            start_symbol=sos_idx,
            end_symbol=eos_idx,
            device=device,
        )

        tgt_mask = make_tgt_mask(decoded, pad_idx=pad_idx).to(device)
        memory = model.encode(src, src_mask)
        _ = model.decode(memory, src_mask, decoded, tgt_mask)

        attn = model.decoder.layers[-1].cross_attn.attention_weights

        if attn is None:
            return

        # Shape: [batch, heads, tgt_len, src_len]
        attn_matrix = attn[0, 0].detach().cpu()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_matrix, aspect="auto")
        ax.set_xlabel("Source positions")
        ax.set_ylabel("Target positions")
        ax.set_title("Decoder Cross-Attention Heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        wandb.log({"attention_heatmap": wandb.Image(fig)})
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    first_encoder_layer = model.encoder.layers[0]

    model_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": first_encoder_layer.self_attn.num_heads,
        "d_ff": first_encoder_layer.ffn.linear1.out_features,
        "dropout": model.positional_encoding.dropout.p,
    }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"])


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    parser = argparse.ArgumentParser(description="Train Transformer on Multi30k")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--smoothing", type=float, default=0.1)

    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="da6401-a3")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb = None
    if args.use_wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    train_dataset = Multi30kDataset(
        split="train",
        min_freq=args.min_freq,
        max_len=args.max_len,
    )

    val_dataset = Multi30kDataset(
        split="validation",
        min_freq=args.min_freq,
        max_len=args.max_len,
    )

    test_dataset = Multi30kDataset(
        split="test",
        min_freq=args.min_freq,
        max_len=args.max_len,
    )

    # Reuse train vocabulary for validation and test to avoid inconsistent token indices.
    val_dataset.src_vocab = train_dataset.src_vocab
    val_dataset.tgt_vocab = train_dataset.tgt_vocab
    val_dataset.src_itos = train_dataset.src_itos
    val_dataset.tgt_itos = train_dataset.tgt_itos
    val_dataset.data = val_dataset.process_data()

    test_dataset.src_vocab = train_dataset.src_vocab
    test_dataset.tgt_vocab = train_dataset.tgt_vocab
    test_dataset.src_itos = train_dataset.src_itos
    test_dataset.tgt_itos = train_dataset.tgt_itos
    test_dataset.data = test_dataset.process_data()

    pad_idx = train_dataset.pad_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx),
    )

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=pad_idx,
        smoothing=args.smoothing,
    )

    if wandb is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.8f}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "train_ppl": math.exp(min(train_loss, 20)),
                    "val_ppl": math.exp(min(val_loss, 20)),
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint_path)

            if wandb is not None:
                wandb.save(args.checkpoint_path)

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_itos,
        device=device,
        max_len=args.max_len,
    )

    print(f"Test BLEU: {bleu:.2f}")

    if wandb is not None:
        wandb.log({"test_bleu": bleu})

        log_sample_translations(
            model=model,
            dataloader=test_loader,
            tgt_vocab=train_dataset.tgt_itos,
            device=device,
            wandb=wandb,
            max_examples=10,
            max_len=args.max_len,
        )

        log_attention_heatmap(
            model=model,
            dataloader=test_loader,
            tgt_vocab=train_dataset.tgt_itos,
            device=device,
            wandb=wandb,
            max_len=args.max_len,
        )

        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()