# Gemma-270M-Model
Trained Small Language Model from Scratch
Gemma-style Transformer (≈270M) — from scratch

A compact, from-scratch implementation of a GPT-style decoder with Grouped Query Attention (GQA), RoPE positional embeddings, RMSNorm everywhere, GEGLU feed-forward, and mixed sliding vs full attention layers. Includes a lean training loop with AMP (bf16/fp16), AdamW, linear warmup + cosine decay, gradient accumulation, memmap’d datasets, and simple text generation.

Highlights

- Architecture: 18 blocks · GQA · RMSNorm · GEGLU · dual RoPE bases (local/global)

- Attention patterns: sliding-window attention in most layers, full attention in selected layers

- RoPE: compute_rope_params + apply_rope with precomputed sin/cos

- Data path: GPT-2 BPE tokenization (tiktoken) → contiguous ids → .bin via np.memmap

- Training: AMP (bf16/fp16), AdamW, warmup→cosine, grad accumulation, best-ckpt saving

- Generation: temperature & top-k sampling
