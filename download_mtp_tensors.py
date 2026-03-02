#!/usr/bin/env python3
"""Download only MTP tensors from Qwen3.5-27B using HTTP range requests.

Instead of downloading 4 full shards (~20GB), this downloads only the MTP
tensor data (~800MB) using the safetensors format's random-access capability.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_url, hf_hub_download

REPO_ID = "Qwen/Qwen3.5-27B"
OUTPUT_DIR = Path(__file__).parent / "models" / "hf"


def get_safetensors_header(url: str, session: requests.Session) -> dict:
    """Download and parse safetensors header using range request."""
    # First 8 bytes = header size (uint64 LE)
    resp = session.get(url, headers={"Range": "bytes=0-7"})
    resp.raise_for_status()
    header_size = struct.unpack("<Q", resp.content)[0]

    # Download header JSON
    resp = session.get(url, headers={"Range": f"bytes=8-{7 + header_size}"})
    resp.raise_for_status()
    header = json.loads(resp.content)
    header["__header_size__"] = header_size
    return header


def download_tensor_data(url: str, session: requests.Session,
                         header_size: int, start: int, end: int) -> bytes:
    """Download specific tensor data using range request."""
    data_offset = 8 + header_size
    byte_start = data_offset + start
    byte_end = data_offset + end - 1  # HTTP range is inclusive
    resp = session.get(url, headers={"Range": f"bytes={byte_start}-{byte_end}"})
    resp.raise_for_status()
    return resp.content


def main():
    # Download index to find MTP tensor locations
    print("Downloading safetensors index...")
    index_path = hf_hub_download(REPO_ID, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Group MTP tensors by shard
    weight_map = index["weight_map"]
    shard_tensors: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        # Match both "mtp." and "model.mtp." prefixes
        clean = tensor_name
        if clean.startswith("model."):
            clean = clean[6:]
        if clean.startswith("mtp."):
            shard_tensors.setdefault(shard_file, []).append(tensor_name)

    print(f"\nMTP tensors spread across {len(shard_tensors)} shards:")
    for shard, tensors in sorted(shard_tensors.items()):
        print(f"  {shard}: {len(tensors)} tensors")

    # Download MTP tensors using range requests
    session = requests.Session()
    all_tensors: dict[str, tuple[np.ndarray, list[int]]] = {}
    total_downloaded = 0

    for shard_file, tensor_names in sorted(shard_tensors.items()):
        url = hf_hub_url(REPO_ID, shard_file)
        print(f"\nProcessing {shard_file}...")

        # Get header
        header = get_safetensors_header(url, session)
        header_size = header.pop("__header_size__")
        # Remove __metadata__ if present
        header.pop("__metadata__", None)

        for tensor_name in tensor_names:
            if tensor_name not in header:
                print(f"  WARNING: {tensor_name} not found in header, skipping")
                continue

            info = header[tensor_name]
            dtype_str = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]
            size_bytes = end - start
            size_mb = size_bytes / 1024 / 1024

            print(f"  Downloading {tensor_name}: {shape} {dtype_str} ({size_mb:.1f} MB)...")
            data = download_tensor_data(url, session, header_size, start, end)
            total_downloaded += len(data)

            # Convert to numpy
            dtype_map = {
                "F32": np.float32,
                "F16": np.float16,
                "BF16": np.dtype("V2"),  # raw bytes for BF16
            }

            if dtype_str == "BF16":
                # Convert BF16 to F32
                raw = np.frombuffer(data, dtype=np.uint16)
                f32 = np.zeros(len(raw), dtype=np.float32)
                f32_view = f32.view(np.uint32)
                f32_view[:] = raw.astype(np.uint32) << 16
                arr = f32.reshape(shape)
            else:
                np_dtype = dtype_map.get(dtype_str)
                if np_dtype is None:
                    print(f"    Unknown dtype {dtype_str}, skipping")
                    continue
                arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)

            # Clean tensor name
            clean_name = tensor_name
            if clean_name.startswith("model."):
                clean_name = clean_name[6:]
            all_tensors[clean_name] = arr
            print(f"    -> {clean_name}: {arr.shape} {arr.dtype}")

    print(f"\nTotal downloaded: {total_downloaded / 1024 / 1024:.1f} MB")
    print(f"Loaded {len(all_tensors)} MTP tensors")

    # Save as safetensors
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors"])
        from safetensors.numpy import save_file

    output_path = OUTPUT_DIR / "mtp_tensors.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # safetensors.numpy.save_file expects dict[str, np.ndarray]
    save_file(all_tensors, str(output_path))
    file_size = output_path.stat().st_size
    print(f"\nSaved: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    # Also copy config.json and create a minimal index pointing to our file
    config_path = hf_hub_download(REPO_ID, "config.json")
    import shutil
    shutil.copy2(config_path, OUTPUT_DIR / "config.json")

    # Create a simple index that points all MTP tensors to our file
    mtp_weight_map = {name: "mtp_tensors.safetensors" for name in all_tensors}
    mini_index = {"metadata": {}, "weight_map": mtp_weight_map}
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(mini_index, f, indent=2)

    print(f"Config and index copied to {OUTPUT_DIR}")
    print("\nReady for injection!")


if __name__ == "__main__":
    main()
