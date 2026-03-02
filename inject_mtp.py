#!/usr/bin/env python3
"""Inject MTP (Multi-Token Prediction) weights into an existing GGUF file.

Takes a pre-quantized GGUF (e.g., unsloth Q4_K_XL) and MTP weight shards
from HuggingFace safetensors, combines them into a new GGUF with MTP support.

Usage:
    python inject_mtp.py <input.gguf> <hf_dir> -o <output.gguf>

    hf_dir should contain:
      - config.json
      - model.safetensors.index.json
      - The safetensors shard(s) containing mtp.* tensors

Example workflow:
    # 1. Download the GGUF
    huggingface-cli download unsloth/Qwen3.5-27B-GGUF \\
        --include "*UD-Q4_K_XL*" --local-dir models/

    # 2. Download HF config + index
    huggingface-cli download Qwen/Qwen3.5-27B \\
        --include "config.json" "model.safetensors.index.json" --local-dir models/hf/

    # 3. Find which shards have MTP weights (this script tells you)
    python inject_mtp.py models/*.gguf models/hf/ -o out.gguf --dry-run

    # 4. Download those shards
    huggingface-cli download Qwen/Qwen3.5-27B \\
        --include "model-00011-of-00011.safetensors" --local-dir models/hf/

    # 5. Inject
    python inject_mtp.py models/*.gguf models/hf/ -o out.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Add the gguf-py package from the local llama.cpp repo
SCRIPT_DIR = Path(__file__).parent
GGUF_PY_DIR = SCRIPT_DIR / "llama.cpp" / "gguf-py"
if GGUF_PY_DIR.exists():
    sys.path.insert(0, str(GGUF_PY_DIR))

import gguf
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType
from gguf.quants import quantize as ggml_quantize


def find_mtp_shards(hf_dir: Path) -> tuple[list[Path], list[str]]:
    """Find safetensors shards containing MTP tensors.

    Returns (existing_shards, missing_shards).
    """
    index_path = hf_dir / "model.safetensors.index.json"
    if not index_path.exists():
        single = hf_dir / "model.safetensors"
        if single.exists():
            return [single], []
        raise FileNotFoundError(f"No safetensors index or model found in {hf_dir}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    mtp_shards = set()
    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith("model.mtp.") or tensor_name.startswith("mtp."):
            mtp_shards.add(shard_file)

    if not mtp_shards:
        raise ValueError("No MTP tensors found in safetensors index")

    existing = []
    missing = []
    for shard in sorted(mtp_shards):
        shard_path = hf_dir / shard
        if shard_path.exists():
            existing.append(shard_path)
        else:
            missing.append(shard)

    return existing, missing


def load_mtp_tensors(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    """Load only MTP tensors from safetensors shards."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors package required. Install: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    mtp_tensors = {}
    for shard_path in shard_paths:
        print(f"  Loading from {shard_path.name}...")
        with safe_open(str(shard_path), framework="numpy") as f:
            for name in f.keys():
                clean_name = name
                if clean_name.startswith("model."):
                    clean_name = clean_name[len("model."):]
                if clean_name.startswith("mtp."):
                    tensor = f.get_tensor(name)
                    mtp_tensors[clean_name] = tensor
                    size_str = f"{tensor.nbytes / 1024:.1f} KB" if tensor.nbytes < 1024*1024 else f"{tensor.nbytes / 1024 / 1024:.1f} MB"
                    print(f"    {clean_name}: {list(tensor.shape)} {tensor.dtype} ({size_str})")

    return mtp_tensors


def remap_mtp_tensor(name: str, n_main: int) -> tuple[str | None, bool]:
    """Remap HF MTP tensor name to GGUF name.

    Returns (gguf_name_template, is_shared).
    is_shared means tensor should be duplicated for all MTP layers.
    """
    # Layer tensors: mtp.layers.{bid}.* -> blk.{bid + n_main}.*
    if name.startswith("mtp.layers."):
        parts = name.split(".", 3)
        bid = int(parts[2])
        rest = parts[3]
        new_bid = bid + n_main

        layer_map = {
            "self_attn.q_proj.weight":  f"blk.{new_bid}.attn_q.weight",
            "self_attn.k_proj.weight":  f"blk.{new_bid}.attn_k.weight",
            "self_attn.v_proj.weight":  f"blk.{new_bid}.attn_v.weight",
            "self_attn.o_proj.weight":  f"blk.{new_bid}.attn_output.weight",
            "self_attn.q_norm.weight":  f"blk.{new_bid}.attn_q_norm.weight",
            "self_attn.k_norm.weight":  f"blk.{new_bid}.attn_k_norm.weight",
            "mlp.gate_proj.weight":     f"blk.{new_bid}.ffn_gate.weight",
            "mlp.up_proj.weight":       f"blk.{new_bid}.ffn_up.weight",
            "mlp.down_proj.weight":     f"blk.{new_bid}.ffn_down.weight",
            "input_layernorm.weight":   f"blk.{new_bid}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{new_bid}.post_attention_norm.weight",
        }

        return layer_map.get(rest), False

    # Shared tensors: same weights for all MTP layers
    shared_map = {
        "mtp.fc.weight":                    "blk.{bid}.nextn.eh_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "blk.{bid}.nextn.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight":    "blk.{bid}.nextn.hnorm.weight",
        "mtp.norm.weight":                  "blk.{bid}.nextn.shared_head_norm.weight",
    }

    return shared_map.get(name), True


def main():
    parser = argparse.ArgumentParser(description="Inject MTP weights into a GGUF file")
    parser.add_argument("input_gguf", type=Path, help="Input GGUF file")
    parser.add_argument("hf_dir", type=Path, help="HF directory with config + MTP safetensors")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output GGUF path")
    parser.add_argument("--mtp-qtype", default="Q8_0",
                        choices=["F32", "F16", "Q8_0"],
                        help="Quantization for MTP weights (default: Q8_0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show which shards are needed, don't inject")
    args = parser.parse_args()

    # Load config
    config_path = args.hf_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in {args.hf_dir}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    n_main = text_config["num_hidden_layers"]
    mtp_layers = text_config.get("mtp_num_hidden_layers", 0)
    hidden_size = text_config["hidden_size"]
    n_total = n_main + mtp_layers

    if mtp_layers == 0:
        print("ERROR: Model has no MTP layers", file=sys.stderr)
        sys.exit(1)

    print(f"Config: {n_main} main + {mtp_layers} MTP = {n_total} total layers, hidden={hidden_size}")

    # Find MTP shards
    print("\nFinding MTP shards...")
    existing_shards, missing_shards = find_mtp_shards(args.hf_dir)

    if missing_shards:
        print(f"\nMissing {len(missing_shards)} shard(s). Download with:")
        include = " ".join(f'"{s}"' for s in missing_shards)
        print(f"  huggingface-cli download Qwen/Qwen3.5-27B --include {include} --local-dir {args.hf_dir}")
        if args.dry_run or not existing_shards:
            return

    if args.dry_run:
        print(f"\nAll {len(existing_shards)} required shard(s) are present. Ready to inject.")
        return

    # Load MTP tensors
    print(f"\nLoading MTP tensors from {len(existing_shards)} shard(s)...")
    mtp_tensors = load_mtp_tensors(existing_shards)
    print(f"Loaded {len(mtp_tensors)} MTP tensors")

    qtype_map = {
        "F32": GGMLQuantizationType.F32,
        "F16": GGMLQuantizationType.F16,
        "Q8_0": GGMLQuantizationType.Q8_0,
    }
    mtp_qtype = qtype_map[args.mtp_qtype]

    # Read original GGUF
    print(f"\nReading input: {args.input_gguf}")
    reader = GGUFReader(str(args.input_gguf))

    arch = "unknown"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = str(bytes(field.parts[-1]), encoding="utf-8")
            break

    print(f"Architecture: {arch}")
    print(f"Original tensors: {len(reader.tensors)}")

    # Prepare MTP tensors for injection
    print(f"\nPreparing MTP tensors ({args.mtp_qtype})...")
    mtp_prepared = []  # list of (name, data_np, raw_shape, raw_dtype_or_None)

    for hf_name, data in mtp_tensors.items():
        gguf_name, is_shared = remap_mtp_tensor(hf_name, n_main)
        if gguf_name is None:
            print(f"  WARNING: Skipping unknown tensor: {hf_name}", file=sys.stderr)
            continue

        data_f32 = data.astype(np.float32)

        if is_shared:
            for mtp_bid in range(n_main, n_total):
                final_name = gguf_name.format(bid=mtp_bid)
                is_small = data.size <= hidden_size * 2

                if is_small:
                    # Norm weights: keep F32
                    mtp_prepared.append((final_name, data_f32, list(data.shape), None))
                    print(f"  {final_name}: {list(data.shape)} F32")
                else:
                    qdata = ggml_quantize(data_f32, mtp_qtype)
                    mtp_prepared.append((final_name, qdata, list(data.shape), mtp_qtype))
                    print(f"  {final_name}: {list(data.shape)} -> {args.mtp_qtype}")
        else:
            # Only apply +1 bias for RMSNorm layer norms (attn_norm, post_attention_norm)
            # NOT for per-head Q/K norms (attn_q_norm, attn_k_norm) which use standard RMSNorm
            is_layer_norm = ("attn_norm" in gguf_name or "post_attention_norm" in gguf_name) and "q_norm" not in gguf_name and "k_norm" not in gguf_name
            is_small = data.size <= hidden_size * 2

            if is_small:
                if is_layer_norm:
                    # Apply Qwen3.5 norm +1 bias
                    data_f32 = data_f32 + 1.0
                    print(f"  {gguf_name}: {list(data.shape)} F32 (norm+1)")
                else:
                    print(f"  {gguf_name}: {list(data.shape)} F32")
                mtp_prepared.append((gguf_name, data_f32, list(data.shape), None))
            else:
                qdata = ggml_quantize(data_f32, mtp_qtype)
                size_mb = qdata.nbytes / 1024 / 1024
                print(f"  {gguf_name}: {list(data.shape)} -> {args.mtp_qtype} ({size_mb:.1f} MB)")
                mtp_prepared.append((gguf_name, qdata, list(data.shape), mtp_qtype))

    total_mtp_bytes = sum(d.nbytes for _, d, _, _ in mtp_prepared)
    print(f"\nTotal MTP data: {total_mtp_bytes / 1024 / 1024:.1f} MB in {len(mtp_prepared)} tensors")

    # Create output writer
    print(f"\nWriting output: {args.output}")
    writer = GGUFWriter(str(args.output), arch)

    # Copy metadata, updating block_count
    block_count_key = f"{arch}.block_count"
    nextn_key = f"{arch}.nextn_predict_layers"

    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == nextn_key:
            continue

        if field.name == block_count_key:
            print(f"  {block_count_key}: {n_main} -> {n_total}")
            writer.add_uint32(block_count_key, n_total)
            continue

        # Copy field value
        ft = field.types[0] if field.types else None
        if ft is None:
            continue

        try:
            parts = field.parts
            if ft == GGUFValueType.STRING:
                writer.add_string(field.name, str(bytes(parts[-1]), encoding="utf-8"))
            elif ft == GGUFValueType.UINT32:
                writer.add_uint32(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.INT32:
                writer.add_int32(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.FLOAT32:
                writer.add_float32(field.name, float(parts[-1][0]))
            elif ft == GGUFValueType.BOOL:
                writer.add_bool(field.name, bool(parts[-1][0]))
            elif ft == GGUFValueType.UINT64:
                writer.add_uint64(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.FLOAT64:
                writer.add_float64(field.name, float(parts[-1][0]))
            elif ft == GGUFValueType.UINT8:
                writer.add_uint8(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.UINT16:
                writer.add_uint16(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.INT16:
                writer.add_int16(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.INT8:
                writer.add_int8(field.name, int(parts[-1][0]))
            elif ft == GGUFValueType.ARRAY:
                # Handle array types - these include tokenizer data, rope sections, etc.
                if len(parts) > 1:
                    arr_type = field.types[-1] if len(field.types) > 1 else None
                    if arr_type == GGUFValueType.STRING:
                        vals = []
                        for idx in field.data:
                            vals.append(str(bytes(parts[idx]), encoding="utf-8"))
                        writer.add_array(field.name, vals)
                    elif arr_type == GGUFValueType.FLOAT32:
                        vals = [float(parts[idx][0]) for idx in field.data]
                        writer.add_array(field.name, vals)
                    elif arr_type == GGUFValueType.INT32:
                        vals = [int(parts[idx][0]) for idx in field.data]
                        writer.add_array(field.name, vals)
                    elif arr_type == GGUFValueType.UINT32:
                        vals = [int(parts[idx][0]) for idx in field.data]
                        writer.add_array(field.name, vals)
                    # Skip other array types
        except Exception as e:
            print(f"  WARNING: Could not copy field '{field.name}': {e}", file=sys.stderr)

    writer.add_uint32(nextn_key, mtp_layers)

    # Add tensor info for all original tensors + MTP tensors
    print("\nAdding tensor info...")

    # Original tensors
    # Note: GGUFReader.shape returns raw GGUF ne[] order, but GGUFWriter expects
    # numpy convention (outermost dimension first), so we must reverse the shape.
    # Also pass np.float32 as dtype to avoid quant_shape_from_byte_shape conversion
    # (it only triggers on uint8 dtype). raw_dtype overrides the GGML type anyway.
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            list(reversed(tensor.shape)),
            np.float32,
            tensor.n_bytes,
            raw_dtype=tensor.tensor_type,
        )

    # MTP tensors - same trick: pass np.float32 to avoid byte shape conversion
    for name, data, raw_shape, raw_dtype in mtp_prepared:
        if raw_dtype is not None:
            writer.add_tensor_info(name, raw_shape, np.float32, data.nbytes, raw_dtype=raw_dtype)
        else:
            writer.add_tensor_info(name, raw_shape, data.dtype, data.nbytes)

    # Write file header + metadata + tensor info
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Write original tensor data
    n_orig = len(reader.tensors)
    print(f"\nWriting {n_orig} original tensors...")
    for i, tensor in enumerate(reader.tensors):
        writer.write_tensor_data(tensor.data)
        if (i + 1) % 50 == 0 or i + 1 == n_orig:
            pct = (i + 1) / n_orig * 100
            print(f"  {i+1}/{n_orig} ({pct:.0f}%)")

    # Write MTP tensor data
    print(f"\nWriting {len(mtp_prepared)} MTP tensors...")
    for name, data, _, _ in mtp_prepared:
        writer.write_tensor_data(data)

    writer.close()

    in_size = args.input_gguf.stat().st_size
    out_size = args.output.stat().st_size
    added = out_size - in_size

    print(f"\nDone!")
    print(f"  Input:  {in_size / 1024**3:.2f} GB ({n_main} layers)")
    print(f"  Output: {out_size / 1024**3:.2f} GB ({n_total} layers)")
    print(f"  Added:  {added / 1024**2:.1f} MB (MTP weights)")
    print(f"\nTest: llama-cli -m {args.output} -p 'Hello' -n 20")


if __name__ == "__main__":
    main()
