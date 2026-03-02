# Technical Report: MTP Speculative Decoding for Qwen3.5-27B

## 1. Background

### 1.1 Model Architecture

Qwen3.5-27B is a 27-billion parameter hybrid language model combining:
- **48 gated-delta-net layers** (recurrent, linear attention)
- **16 full attention layers** (standard transformer, every 4th layer)
- **1 MTP layer** (Multi-Token Prediction, layer index 64)

Key dimensions:
- `n_embd = 5120`
- `n_head = 24`, `n_head_kv = 4` (GQA)
- `n_embd_head_v = 256` (note: larger than the usual 128)
- `n_vocab = 248320`

The recurrent layers maintain a gated-delta-net state (~150 MiB total across 48 layers), which creates unique challenges for speculative decoding.

### 1.2 MTP Layer Architecture

The MTP layer predicts the *next* token given:
1. The main model's pre-normalization hidden state at the last output position
2. The embedding of the most recently sampled token

```
                 ┌─────────────────┐
hidden_state ──→ │ hnorm (RMSNorm) │──┐
                 └─────────────────┘  │    ┌────────┐    ┌─────────────────────┐
                                      ├──→ │ concat │──→ │ eh_proj (5120→5120) │
                 ┌─────────────────┐  │    └────────┘    └──────────┬──────────┘
tok_embedding ──→│ enorm (RMSNorm) │──┘                             │
                 └─────────────────┘                                ▼
                                                    ┌───────────────────────────┐
                                                    │ Transformer Block         │
                                                    │ • Gated attention (24 hd) │
                                                    │ • SwiGLU FFN              │
                                                    └──────────┬────────────────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────┐
                                                    │ shared_head_norm │
                                                    │ shared_head_head │──→ logits [248320]
                                                    └──────────────────┘
```

## 2. Implementation

### 2.1 GGUF Weight Preparation

**Problem**: Pre-quantized GGUFs (e.g., from Unsloth) don't include MTP weights because llama.cpp didn't support them.

**Solution**: Two Python utilities:

1. **`download_mtp_tensors.py`** - Uses HTTP range requests against the HuggingFace safetensors format to download only the MTP tensor data (~800MB) instead of full model shards (~20GB).

2. **`inject_mtp.py`** - Reads MTP tensors from safetensors, quantizes them to match the base GGUF's quantization, and appends them to a copy of the GGUF. Also patches metadata (`nextn_predict_layers`, tensor count).

MTP tensors and their GGUF mappings:
| HuggingFace Name | GGUF Name | Shape | Purpose |
|---|---|---|---|
| `mtp.0.fc.weight` | `blk.64.nextn.eh_proj.weight` | [5120, 10240] | Projection after concat |
| `mtp.0.pre_fc_norm_embedding.weight` | `blk.64.nextn.enorm.weight` | [5120] | Embedding norm |
| `mtp.0.pre_fc_norm_hidden.weight` | `blk.64.nextn.hnorm.weight` | [5120] | Hidden state norm |
| `mtp.0.block.attn_norm.weight` | `blk.64.attn_norm.weight` | [5120] | Pre-attention norm |
| `mtp.0.block.{wq,wk,wv,wo}.weight` | `blk.64.attn_{q,k,v,output}.weight` | various | Attention weights |
| `mtp.0.block.ffn_{gate,up,down}.weight` | `blk.64.ffn_{gate,up,down}.weight` | various | FFN weights |
| `mtp.0.block.attn_post_norm.weight` | `blk.64.attn_post_norm.weight` | [5120] | Post-attention norm |
| `mtp.0.shared_head.norm.weight` | `blk.64.nextn.shared_head_norm.weight` | [5120] | Output norm |
| `mtp.0.shared_head.head.weight` | `blk.64.nextn.shared_head_head.weight` | [248320, 5120] | Output projection |

### 2.2 Hidden State Extraction

The MTP layer needs the main model's hidden state *before* the final RMSNorm. This is captured by:

1. Adding `t_embd_prenorm` to `llm_graph_result` (graph output alongside `t_embd`)
2. In `qwen35.cpp`, saving `cur` before the final norm: `res->t_embd_prenorm = cur`
3. After main model decode, copying the last output position's hidden state to `mtp_hidden_state` vector

### 2.3 MTP Forward Pass (`decode_mtp()`)

The MTP forward pass builds a fresh ggml graph each call (~30 nodes, ~microseconds to construct). Key implementation details:

**Norm +1 offset**: The GGUF converter adds `+1` to tensors matching `"*.norm.weight"`, implementing the `(1 + w) * rmsnorm(x)` convention. But `enorm.weight` and `hnorm.weight` don't match this pattern, so their values are ~[-0.56, -0.26] instead of ~[0.44, 0.74]. Fixed at runtime:
```cpp
ggml_tensor * hnorm_plus1 = ggml_add1(ctx0, layer.nextn.hnorm, mtp_inp_one);
h_normed = ggml_mul(ctx0, h_normed, hnorm_plus1);
```

**Concat order**: Determined empirically and confirmed against vLLM's DeepSeek MTP reference:
```cpp
// [embed, hidden] - embeddings first, matching vLLM's torch.cat([inputs_embeds, hidden], dim=-1)
ggml_tensor * concat = ggml_concat(ctx0, e_normed, h_normed, 0);
```

**Single-token attention simplification**: For a single token, the attention mechanism simplifies significantly:
- `softmax(QK^T / sqrt(d))` = `softmax(scalar)` = 1.0
- So `attn_output = V` (the value vector itself)
- With gating: `attn_output = V * sigmoid(gate)`
- The gate is extracted from the Q projection (which outputs `n_head * n_embd_head * 2`, where the second half is the gate)

**Head count derivation**: Instead of using `hparams.n_head()` (which might differ from the MTP layer), dimensions are derived from weight shapes:
```cpp
const int64_t mtp_n_head    = layer.wq->ne[1] / (n_embd_head * 2);  // /2 for gating
const int64_t mtp_n_head_kv = layer.wv->ne[1] / n_embd_head;
```

**Dedicated scheduler**: The MTP graph uses its own `ggml_backend_sched` to avoid disturbing the main model's scheduler state. Using the main scheduler caused crashes or expensive re-reservation.

### 2.4 Speculative Decoding Integration

**Draft generation** (`common/speculative.cpp`):
1. Call `llama_decode_mtp(ctx, sampled_token)`
2. Get logits via `llama_get_logits_mtp(ctx)`
3. Greedy argmax over `n_vocab` logits
4. Return 1 draft token

**Server-side verification** (`server-context.cpp`):
1. After sampling a token, save recurrent state checkpoint
2. Build a batch with `[sampled_token, draft_token]`
3. Decode the batch (verification)
4. If draft matches verification: accept both tokens
5. If draft rejected: restore checkpoint, defer correction tokens to next batch

**Deferred correction**: Instead of immediately re-decoding the correct tokens (expensive extra forward pass), rejected speculation stores the correction tokens and prepends them to the next decode batch. This saves one forward pass per rejection.

### 2.5 GPU Checkpoint System

The recurrent state checkpoint was the single biggest overhead. Evolution:

| Approach | Mechanism | Save+Restore Time |
|---|---|---|
| CPU checkpoint | `ggml_backend_tensor_get/set` (sync, PCIe) | ~42ms |
| GPU checkpoint (with sync) | `ggml_backend_tensor_copy_async` + `ggml_backend_synchronize` | ~20ms |
| GPU checkpoint (no sync) | `ggml_backend_tensor_copy_async` only, relying on CUDA stream ordering | ~20ms |

The GPU checkpoint allocates shadow tensors on the same GPU device as the source tensors and uses `ggml_backend_tensor_copy_async()` which resolves to `cudaMemcpyAsync` for same-device copies. This avoids the PCIe bottleneck entirely.

Implementation details:
- `checkpoint_alloc()`: Groups layers by buffer type, creates matching checkpoint tensors
- `set_checkpoint_backends()`: Receives backend handles from the context (needed for async copy API)
- `find_backend_for_tensor()`: Maps tensor → buffer type → backend handle
- Stream ordering: No explicit sync needed since copies and subsequent graph compute share the same backend stream (CUDA FIFO ordering)

## 3. Results

### 3.1 Acceptance Rate

Over a 200-token generation run:
- **99 MTP draft calls**
- **47 accepted** (47.5% acceptance rate)
- Acceptance is bursty: streaks of 3-4 acceptances followed by streaks of 5-6 rejections

### 3.2 Performance

| Configuration | Speed | vs Baseline |
|---|---|---|
| Baseline (no MTP) | ~17 t/s | — |
| MTP (GPU checkpoint, no sync) | ~12.5 t/s | -26% |

The MTP overhead per cycle:
- MTP graph compute: ~4.8ms
- Checkpoint save: ~10ms (150 MiB GPU-to-GPU)
- Checkpoint restore (on rejection): ~10ms
- Verification batch overhead: ~15ms (2 tokens vs 1 for recurrent layers)

### 3.3 Cost Analysis

**Normal decode**: ~59ms per token → 17 t/s

**MTP cycle** (produces 1.475 tokens on average):
- Always: 59ms (verify batch) + 4.8ms (MTP) + 10ms (checkpoint save) = 73.8ms
- On rejection (52.5%): +10ms (restore) = 5.25ms average
- Total: ~79ms for 1.475 tokens → ~18.7 t/s theoretical

But the verification batch takes longer than single-token decode for recurrent models (~74ms vs 59ms for 2 tokens), making the actual effective rate ~12.5 t/s.

## 4. Challenges & Debugging History

### 4.1 All-Zero Logits (Separate Scheduler)
**Symptom**: MTP graph with a dedicated scheduler produced all-zero logits.
**Cause**: Unknown (possibly buffer allocation issue with the first approach).
**Fix**: Rebuild the entire graph fresh each call instead of caching it.

### 4.2 Crash on Second MTP Call (Stale Tensors)
**Symptom**: `GGML_ASSERT(backend_res != nullptr)` when reusing a cached MTP graph.
**Cause**: The graph's tensor buffer pointers became stale after the main scheduler reallocated buffers.
**Fix**: Rebuild the ggml context and graph from scratch each `decode_mtp()` call. Construction cost is microseconds, negligible vs ~5ms compute.

### 4.3 Zero Acceptance Rate (Wrong Predictions)
**Root cause 1**: enorm/hnorm missing `+1` offset. Values were ~[-0.56, -0.26] instead of ~[0.44, 0.74]. The GGUF converter's norm pattern didn't match these tensor names.

**Root cause 2**: Concat order was `[hidden, embed]` but should be `[embed, hidden]`. Confirmed by A/B test and vLLM reference.

**Root cause 3** (minor): Initially assumed `n_embd_head = 128` but the actual value is 256 for this model, affecting attention head count calculations.

### 4.4 Crash After First Acceptance (Main Scheduler Corruption)
**Symptom**: Crash in `decode()` after successful MTP → verify cycle.
**Cause**: MTP computation with the main scheduler corrupted its state for the next decode.
**Fix**: Use a dedicated `mtp_sched` for the MTP graph.

### 4.5 Slow Performance with Main Scheduler
**Symptom**: 8.2 t/s when using main scheduler for MTP (adding `sched_need_reserve = true` after MTP).
**Cause**: Scheduler re-reservation is expensive (~70ms), wiping out any speculative benefit.
**Fix**: Dedicated MTP scheduler (no re-reservation needed).

## 5. Conclusions

### What Works
- MTP forward pass produces correct logits matching the model's predictions ~47.5% of the time
- GPU-to-GPU checkpoint reduces state transfer overhead significantly
- Deferred correction avoids extra forward passes on rejection
- The implementation is lossless (speculative decoding with greedy verification)

### What Doesn't Work (Yet)
- **Net throughput is negative**: MTP is ~26% slower than baseline due to the fundamental cost structure of speculative decoding on hybrid recurrent models
- The recurrent state creates an inherent sequential dependency that limits batch verification speedups
- 47.5% acceptance is too low to overcome the overhead

### Potential Future Improvements
1. **Higher acceptance rate**: Fine-tuning the MTP layer, or using non-greedy draft sampling with temperature
2. **Multi-token drafting**: Chain multiple MTP calls (predict 2-3 tokens ahead) to amortize checkpoint cost over more tokens
3. **Checkpoint-free speculation**: For models where recurrent state update is cheap/reversible
4. **Selective checkpointing**: Only checkpoint layers whose state actually changes (all recurrent layers change, so limited benefit here)
5. **Async pipeline**: Overlap MTP compute with checkpoint save
6. **Fix enorm/hnorm in GGUF converter**: Add these tensor names to the norm `+1` offset pattern so the fix isn't needed at runtime
7. **Attention-only models**: MTP would likely be much more beneficial for pure-attention models where batch verification is nearly free and there's no recurrent state to checkpoint
