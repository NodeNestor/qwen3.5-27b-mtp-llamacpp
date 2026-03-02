# Qwen3.5-27B MTP Speculative Decoding for llama.cpp

An experimental implementation of **Multi-Token Prediction (MTP)** speculative decoding for [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) in [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Motivation

Qwen3.5-27B ships with a built-in MTP (Multi-Token Prediction) head - a single extra transformer layer trained to predict the next token from the main model's hidden state. This is designed for speculative decoding: the MTP layer drafts a token cheaply, then the main model verifies it. When the draft is correct, you get two tokens for roughly the cost of one forward pass. No separate draft model needed.

The problem: **llama.cpp didn't implement the MTP forward pass**. The weights were loaded but unused. And the pre-quantized GGUFs on HuggingFace didn't even include the MTP weights.

This project set out to:
1. Get the MTP weights into a usable GGUF format
2. Implement the MTP forward pass in llama.cpp's C++ inference engine
3. Wire it into the speculative decoding pipeline
4. See if it actually makes inference faster

**TL;DR**: The MTP layer works correctly (~47.5% acceptance rate), but speculative decoding is a net negative for this model because the hybrid recurrent architecture makes checkpoint/restore too expensive. See [Performance Results](#performance-results) and [REPORT.md](REPORT.md) for details.

## What is MTP?

Qwen3.5-27B is a hybrid architecture with 64 main layers (48 gated-delta-net recurrent + 16 full attention) and **1 MTP layer** (layer index 64). The MTP layer is a single transformer block that takes:
- The main model's pre-norm hidden state
- The embedding of the sampled token

...and predicts what the *next* token should be. In theory, this enables speculative decoding using only the model itself.

## What This Fork Does

### Step 1: Getting MTP Weights into GGUF

Pre-quantized GGUFs on HuggingFace (e.g., Unsloth's Q4_K_M) don't include MTP weights. Re-quantizing the full 27B model just to add one layer is wasteful. Instead:

- **`download_mtp_tensors.py`** - Downloads only the MTP tensor data (~800MB) from HuggingFace using HTTP range requests against the safetensors format. This exploits safetensors' random-access design to avoid downloading the full ~20GB shards.
- **`inject_mtp.py`** - Takes a pre-quantized GGUF and the downloaded MTP tensors, quantizes the MTP weights to match, and produces a new GGUF with MTP support. Patches the metadata (`nextn_predict_layers`, tensor count) so llama.cpp recognizes the MTP layer.
- **GGUF converter patches** (`convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py`) - Adds tensor name mappings for MTP-specific weights: `eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, `shared_head_head`.

### Step 2: MTP Forward Pass (C++)
The core contribution: a complete MTP forward pass implementation in llama.cpp's inference engine.

**Modified files** (948 lines added across 18 files):

| File | Changes |
|------|---------|
| `src/llama-context.cpp` | MTP graph construction, `decode_mtp()`, hidden state extraction |
| `src/llama-context.h` | MTP state members (hidden state, logits, scheduler, graph tensors) |
| `src/llama-graph.cpp/h` | `t_embd_prenorm` output for saving pre-norm hidden state |
| `src/models/qwen35.cpp` | Saves pre-norm hidden state during forward pass |
| `common/speculative.cpp` | MTP draft implementation (`common_speculative_state_mtp`) |
| `include/llama.h` | Public API: `llama_decode_mtp`, `llama_get_logits_mtp`, etc. |
| `src/llama-memory-recurrent.cpp/h` | GPU-to-GPU checkpoint save/restore for recurrent state |
| `src/llama-memory-hybrid.cpp/h` | Checkpoint delegation to recurrent memory |
| `tools/server/server-context.cpp` | Deferred correction on draft rejection |
| `src/llama-model.cpp` | MTP weight loading and layer configuration |
| `src/llama-arch.cpp` | Architecture tensor name mappings |

### Key Technical Details

**MTP Architecture** (single-token inference):
```
hidden_state (from main model, pre-norm) ──→ RMSNorm(hnorm + 1) ──┐
                                                                    ├──→ concat ──→ eh_proj ──→ [n_embd]
token_embedding (of sampled token)       ──→ RMSNorm(enorm + 1) ──┘

──→ Transformer Block (attn + SwiGLU FFN) ──→ shared_head_norm ──→ shared_head_head ──→ logits
```

**Discoveries during implementation:**
- **Norm +1 offset**: `enorm` and `hnorm` weights require a `+1` offset (`(1 + w) * rmsnorm(x)`) that the GGUF converter misses because their names don't match the `"norm.weight"` suffix pattern. Fixed at runtime with `ggml_add1()`.
- **Concat order**: Must be `[embed, hidden]` (not `[hidden, embed]`), matching the vLLM/DeepSeek reference implementation.
- **Single-token attention simplification**: For 1 token, `softmax(QK^T/sqrt(d)) = 1.0`, so attention reduces to `Wo * (V * sigmoid(gate))`. The gated Q projection extracts the gate signal.
- **Head dimensions**: MTP layer uses `n_embd_head_v = 256` (not 128), with `n_head = 24` and `n_head_kv = 4` matching the main model.

**GPU Checkpoint System:**
For hybrid recurrent models, speculative decoding requires saving/restoring the recurrent state on draft rejection. This fork implements:
- GPU-side checkpoint tensors allocated on the same device as source tensors
- `ggml_backend_tensor_copy_async()` for device-to-device memcpy (no CPU roundtrip)
- Zero-sync design relying on CUDA stream ordering

## Performance Results

Tested with Qwen3.5-27B Q4_K_M on RTX 4060 (8GB) + RTX 5060 Ti (16GB), tensor split 8:16.

| Configuration | Generation Speed | Notes |
|---|---|---|
| **Baseline (no MTP)** | **~17 t/s** | Standard single-token decode |
| MTP (CPU checkpoint) | ~11.7 t/s | GPU→CPU→GPU state transfer |
| MTP (GPU checkpoint) | ~12.5 t/s | GPU→GPU async copies |
| MTP (GPU, no sync) | ~12.5 t/s | Removing sync barriers didn't help further |

**Draft acceptance rate: ~47.5%** (47 accepted / 99 generated in a 200-token run)

### Why MTP Is Currently Slower

MTP speculative decoding is a net negative for this model/hardware combination:

1. **Recurrent state overhead**: The 48 gated-delta-net layers maintain ~150 MiB of recurrent state that must be checkpointed before speculation and restored on rejection.
2. **Low acceptance rate**: At ~47.5%, over half of speculation cycles are pure overhead (checkpoint + MTP compute + restore + deferred correction).
3. **Recurrent batch penalty**: Unlike attention-only models where batch verification is nearly free, recurrent layers must process tokens sequentially. A 2-token verification batch takes ~1.75x (not 1x) the time of a single token.
4. **MTP compute cost**: Each MTP call adds ~4.8ms of GPU compute for the small transformer graph.

**For MTP to break even on this architecture, the acceptance rate would need to be >70%, or checkpoint overhead would need to be near-zero.**

## Usage

### Building

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
```

### Creating an MTP-enabled GGUF

```bash
# Download a pre-quantized GGUF (no MTP weights)
huggingface-cli download unsloth/Qwen3.5-27B-GGUF \
    --include "*Q4_K_M*" --local-dir models/

# Download MTP tensors efficiently (~800MB instead of ~20GB)
python download_mtp_tensors.py

# Inject MTP weights into the GGUF
python inject_mtp.py models/Qwen3.5-27B-Q4_K_M.gguf models/hf/ \
    -o models/Qwen3.5-27B-Q4_K_M-MTP.gguf
```

### Running with MTP

```bash
# CLI with MTP speculative decoding
llama-cli -m models/Qwen3.5-27B-Q4_K_M-MTP.gguf \
    -p "Hello" -n 100 \
    --spec-type mtp --draft-max 1 \
    -ngl 99

# Without MTP (baseline)
llama-cli -m models/Qwen3.5-27B-Q4_K_M-MTP.gguf \
    -p "Hello" -n 100 \
    -ngl 99
```

### Server mode

```bash
llama-server -m models/Qwen3.5-27B-Q4_K_M-MTP.gguf \
    --spec-type mtp --draft-max 1 \
    -ngl 99
```

## File Overview

```
.
├── README.md                    # This file
├── REPORT.md                    # Detailed technical report
├── inject_mtp.py                # Inject MTP weights into GGUF
├── download_mtp_tensors.py      # Download only MTP tensors from HF
├── llama.cpp/                   # Modified llama.cpp (based on ecbcb7ea9)
│   ├── src/
│   │   ├── llama-context.cpp    # MTP decode_mtp() implementation
│   │   ├── llama-context.h      # MTP state declarations
│   │   ├── llama-graph.cpp/h    # Pre-norm hidden state output
│   │   ├── llama-memory-recurrent.cpp/h  # GPU checkpoint system
│   │   ├── llama-model.cpp      # MTP weight loading
│   │   └── models/qwen35.cpp    # Hidden state extraction
│   ├── common/speculative.cpp   # MTP draft generation
│   ├── include/llama.h          # Public MTP API
│   └── tools/server/            # Server with deferred correction
└── models/                      # Model files (not tracked)
```

## Limitations

- **Single draft token only** (`--draft-max 1`). Multi-token chaining is implemented but untested.
- **Greedy sampling only** for draft tokens (argmax over MTP logits).
- **enorm/hnorm +1 fix** is applied at runtime, not in the GGUF converter. A proper fix would add these to the converter's norm offset pattern.
- **No CUDA graph support** for the MTP forward pass (graph is rebuilt each call, ~microseconds overhead).
- Output quality is identical to baseline (speculative decoding is lossless with greedy verification).

## Based On

- **llama.cpp**: commit `ecbcb7ea9` (upstream)
- **Model**: [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- **MTP reference**: [vLLM DeepSeek MTP implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py)

## License

The llama.cpp modifications follow llama.cpp's MIT license. The utility scripts (`inject_mtp.py`, `download_mtp_tensors.py`) are provided as-is.
