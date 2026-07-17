#!/usr/bin/env python
"""Non-destructive GPU preflight for the benchmark."""
import argparse
import json
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    args = parser.parse_args()
    result = {"cuda_visible": None, "gpus": []}
    try:
        import torch

        result["cuda_visible"] = bool(torch.cuda.is_available())
        if result["cuda_visible"]:
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                result["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i), "free_gib": free / 2**30, "total_gib": total / 2**30})
    except Exception as exc:
        result["torch_error"] = repr(exc)
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True)
        result["nvidia_smi"] = [line.strip() for line in output.splitlines() if line.strip()]
    except Exception as exc:
        result["nvidia_smi_error"] = repr(exc)
    print(json.dumps(result, indent=2))
    if not result["cuda_visible"] or not any(g["free_gib"] >= args.min_free_gib for g in result["gpus"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
