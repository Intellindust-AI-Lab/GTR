"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Extract the inner VisionTransformer state_dict from a trained GTR checkpoint.

Output is a raw state_dict compatible with `ViTAdapter._load_weights`, i.e. it can
be dropped into the YAML field:

    ViTAdapter:
      weights_path: /path/to/extracted_backbone.pth

Works for s / m / l / x — only the checkpoint differs.

Usage:
    python tools/extract_backbone.py \
        --ckpt outputs/coco_finetune/gtr_s/checkpoint0075.pth \
        --out  outputs/coco_finetune/gtr_s/backbone_ep75_ema.pth

    # use the non-EMA weights instead
    python tools/extract_backbone.py --ckpt ... --out ... --source model
"""

import argparse
from pathlib import Path

import torch


PREFIX = "backbone.backbone._model."


def extract(ckpt_path: Path, out_path: Path, source: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if source == "ema":
        if "ema" not in ckpt or ckpt["ema"] is None:
            raise KeyError(f"No 'ema' entry in {ckpt_path}; pass --source model")
        full_sd = ckpt["ema"]["module"]
    elif source == "model":
        full_sd = ckpt["model"]
    else:
        raise ValueError(f"--source must be ema|model, got {source}")

    backbone_sd = {
        k[len(PREFIX):]: v
        for k, v in full_sd.items()
        if k.startswith(PREFIX)
    }
    if not backbone_sd:
        raise RuntimeError(
            f"No keys with prefix '{PREFIX}' found in checkpoint. "
            f"Sample keys: {list(full_sd.keys())[:5]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_sd, out_path)

    print(f"[extract_backbone] source       : {source}")
    print(f"[extract_backbone] input ckpt   : {ckpt_path}")
    print(f"[extract_backbone] output file  : {out_path}")
    print(f"[extract_backbone] num tensors  : {len(backbone_sd)}")
    print(f"[extract_backbone] total params : {sum(v.numel() for v in backbone_sd.values()):,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source", choices=["ema", "model"], default="ema")
    args = p.parse_args()
    extract(args.ckpt, args.out, args.source)


if __name__ == "__main__":
    main()
