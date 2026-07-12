"""End-to-end smoke test of the SFM Denoise fine-tuning pipeline on the GPU.

Proves that, on this Blackwell GPU with PyTorch 2.11 + timm 1.0.27 and the
torch._six compat shim (sitecustomize.py), the full training code path runs:
model build -> forward -> loss -> backward -> optimizer step -> eval.

It uses a tiny synthetic in-memory dataset (NOT the 8 GB DenoiseSet download),
so it validates the *code path*, not model quality.

Run:
    PYTHONPATH=sandbox/SFM uv run python sandbox/SFM/smoke_denoise.py
"""
import os
import sys
import argparse

import torch
import numpy as np

# Make the SFM-Finetune package importable (models_*, engine_finetune, util.*)
SFM_FT = os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "seismicfoundationmodel", "SFM-Finetune"
)
SFM_FT = os.path.abspath(SFM_FT)
sys.path.insert(0, SFM_FT)

import models_Regression
from models_Regression import forward_loss
from engine_finetune import train_one_epoch, evaluateRegression
from util.misc import NativeScalerWithGradNormCount as NativeScaler


class TinyDenoise(torch.utils.data.Dataset):
    """Mimics DenoiseSet output: (noisy [1,H,W], clean [1,H,W]) float32 tensors."""

    def __init__(self, n=4, size=224):
        self.n, self.size = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        clean = np.random.randn(1, self.size, self.size).astype(np.float32)
        noisy = (clean + 0.3 * np.random.randn(1, self.size, self.size)).astype(np.float32)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))

    # Exactly how main_finetune.py builds the Denoise model.
    model = models_Regression.__dict__["vit_base_patch16"](
        img_size=224, num_classes=1, drop_path_rate=0.1, in_chans=1, Interpolation=False
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built: vit_base_patch16, {n_params/1e6:.1f}M params")

    ds = TinyDenoise(n=4, size=224)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_scaler = NativeScaler()

    # Minimal args object matching what train_one_epoch reads.
    args = argparse.Namespace(
        accum_iter=1, clip_grad=None, lr=1e-4, min_lr=1e-6,
        epochs=1, warmup_epochs=0, blr=1e-4,
    )

    print("Running 1 training epoch on GPU ...")
    stats = train_one_epoch(
        model, forward_loss, loader, optimizer, device, 0, loss_scaler,
        0, None, log_writer=None, task="Denoise", args=args,
    )
    print("train stats:", stats)

    print("Running regression eval ...")
    eval_stats = evaluateRegression(loader, model, device, task="Denoise")
    print("eval stats:", eval_stats)

    print("\nSMOKE TEST PASSED: denoise fine-tuning code path runs end-to-end on GPU.")


if __name__ == "__main__":
    main()
