---
name: seismicfoundationmodel
description: "Use when you need a repository-specific analysis of seismicfoundationmodel (Seismic Foundation Model, SFM): summarize its purpose, explain each important script/module/function, describe execution flow, and show how to run it. SFM is a Masked-Autoencoder ViT foundation model for seismic images, pre-trained self-supervised on 2.28M 224x224 seismic patches and fine-tuned for facies classification, geobody/salt segmentation, denoising, reflectivity inversion, and interpolation."
---

# SeismicFoundationModel (SFM) Repository Analysis

## Purpose

`seismicfoundationmodel` (SFM) is the official PyTorch/GPU implementation of
*"Seismic Foundation Model (SFM): a new generation deep learning model in geophysics"*
(Sheng, Wu, Si, Li, Zhang, Duan — USTC & Huawei, arXiv:2309.02791, accepted by *Geophysics* 2024).

It is a **direct fork/modification of Meta's MAE** (Masked Autoencoder) repo. The idea:

1. **Pre-train** a Vision Transformer (ViT) encoder self-supervised on ~2.28M unlabeled 224×224
   seismic image patches by masking 75% of patches and reconstructing them (MAE objective).
2. **Fine-tune** that pre-trained encoder as a backbone for five downstream seismic tasks:
   - **Facies classification** (`SEAM` task, 6 classes, 768×768) — segmentation head
   - **Geobody / Salt identification** (`Salt` task, 2 classes, 224×224) — segmentation head
   - **Denoising** (`Denoise` task, regression, 224×224)
   - **Reflectivity estimation / Inversion** (`Reflection` task, regression, 224×224)
   - **Interpolation** (trace reconstruction, regression, 224×224, masked-reconstruction style)

The claim is that a single self-supervised pre-trained backbone transfers to all of these,
outperforming training from scratch and matching/beating task-specific U-Net / DeepLab baselines
(both of which are bundled for comparison).

## Verified Repository Summary

Top-level layout (verified by direct inspection of the clone at
`tools/seismicfoundationmodel/`):

```
seismicfoundationmodel/
├── README.md                # Overview, model zoo, data links, install & quick guide
├── requirements.txt         # timm==0.3.2, scikit-learn, matplotlib, tensorboard
├── assert/                  # Figures used in the README (Network.png, SeismicPretrainedModel.png)
├── Data/                    # Per-task dataset READMEs ONLY (no actual data shipped)
│   ├── README-Pretrain.md   #  2,286,422 × 224×224 float32 patches (split zip)
│   ├── README-Facies.md     #  117 × 768×768 (100 train / 17 val)
│   ├── README-Geobody.md    #  4000 × 224×224 (3500 train / 500 val)  -> Salt task
│   ├── README-Denoise.md    #  2000 train pairs + 4000 "field" val, 224×224
│   ├── README-Inversion.md  #  2200 train pairs + 5000 SEAM val, 224×224 -> Reflection task
│   └── README-Interpolation.md # 8000 × 224×224 (6000 train "<n>.dat" / 2000 val "U<n>.dat")
├── SFM-Pretrain/            # Self-supervised MAE pre-training
│   ├── main_pretrain.py     # Entrypoint (argparse + train loop)
│   ├── models_mae.py        # MaskedAutoencoderViT + arch factory funcs
│   ├── engine_pretrain.py   # train_one_epoch (MAE reconstruction loss)
│   ├── submitit_pretrain.py # SLURM/submitit multi-node launcher
│   ├── train.sh             # Single-node launch example
│   ├── submit-train.sh / slurmjob.sh  # Multi-node launch examples
│   └── util/                # datasets, misc, lr_sched, pos_embed, msssim, metrics ...
└── SFM-Finetune/            # Supervised fine-tuning for downstream tasks
    ├── main_finetune.py     # Entrypoint (task dispatch, model build, train/eval loop)
    ├── models_Segmentation.py # ViT backbone + MLA/Cup decoder for classification/segmentation
    ├── models_Regression.py   # ViT backbone + decoder + regression loss (denoise/invert/interp)
    ├── models_mae.py          # (copy of MAE model, present in finetune dir too)
    ├── engine_finetune.py     # train_one_epoch, evaluate (acc/miou), evaluateRegression (mse/msssim/psnr)
    ├── finetune-Facies.sh     # Facies (SEAM) launch example
    ├── Application/           # Per-task launch scripts + visualization.ipynb
    │   ├── finetune-Denoise.sh / finetune-Salt.sh / finetune-Interpolation.sh / finetune-Reflect.sh
    │   └── visualization.ipynb
    ├── modules/modeling/      # Bundled DeepLab + U-Net baselines (for --modelComparsion)
    │   ├── deeplab.py, aspp.py, decoder.py, Unet_models.py
    │   ├── backbone/ (resnet, xception, drn, mobilenet)
    │   └── sync_batchnorm/
    └── util/                  # datasets.py (all Dataset classes), lr_decay, misc, msssim, metrics, tools, pos_embed
```

**Key verified facts / constraints:**

- **No data and no model weights are checked in.** `Data/` holds only README stubs with
  download links (USTC `rec.ustc.edu.cn` + Baidu Netdisk). Pre-trained checkpoints
  (SFM-Base/Large, 224 & 512) are external downloads listed in the top `README.md` "Model Zoo".
- **Framework:** PyTorch (`torch==1.8.1+cu111` recommended in README) + **`timm==0.3.2`**
  (a hard `assert timm.__version__ == "0.3.2"` in `main_pretrain.py`). timm 0.3.2 needs a
  known one-line patch to import under PyTorch 1.8+.
- **All I/O is raw `float32` binary `.dat` files** read via `np.fromfile(...).reshape(...)`.
  There is NO SEGY handling in this repo — inputs are pre-extracted image patches.
- Pinned Python in README: `conda create -n SFM python=3.9.12`.
- Two licenses coexist: top README says MIT; the sub-READMEs (`SFM-Pretrain`, `SFM-Finetune`)
  say **CC-BY-NC 4.0** (non-commercial). Treat downstream code as non-commercial.

## Repository Walkthrough

### `SFM-Pretrain/models_mae.py` — the MAE model

Defines `class MaskedAutoencoderViT(nn.Module)`, an asymmetric encoder–decoder ViT.

#### Function-by-function analysis

- `__init__(img_size=224, patch_size=16, in_chans=1, embed_dim, depth, num_heads, decoder_*)`
  Builds: `PatchEmbed` (from timm) → fixed sin-cos `pos_embed` → `cls_token` → `depth` transformer
  `Block`s → encoder `norm`. Decoder: `decoder_embed` linear → learned `mask_token` → decoder
  `pos_embed` → `decoder_depth` `Block`s → `decoder_pred` linear projecting back to
  `patch_size**2 * in_chans` pixels. **Note `in_chans=1`** — seismic is single-channel; the
  pretrain entrypoint passes `in_chans=1` explicitly.
- `initialize_weights()` — copies 2D sin-cos positional embeddings (from `util.pos_embed.get_2d_sincos_pos_embed`),
  xavier-inits the patch-embed conv as if linear, normal-inits cls/mask tokens.
- `patchify(imgs)` / `unpatchify(x)` — convert between image tensors and flattened patch-token
  sequences. (`unpatchify` is hard-coded to 3 channels — a latent bug, but unused on the loss path
  which uses `patchify` with `self.in_chans`.)
- `random_masking(x, mask_ratio)` — the core MAE trick: per-sample random shuffle via
  `argsort(noise)`, keep the first `L*(1-mask_ratio)` tokens, return the kept subset, a binary
  `mask` (0 keep / 1 remove), and `ids_restore` to unshuffle later.
- `forward_encoder(x, mask_ratio)` — patch-embed → add pos-embed → `random_masking` → prepend
  cls token → transformer blocks → norm. Only ~25% of patches flow through the (heavy) encoder.
- `forward_decoder(x, ids_restore)` — project to decoder dim, re-insert `mask_token`s at the
  removed positions, unshuffle with `ids_restore`, add decoder pos-embed, run decoder blocks,
  project each token to pixels, drop cls token.
- `forward_loss(imgs, pred, mask)` — MSE between predicted and true patch pixels, **averaged only
  over the removed (masked) patches**. Optional per-patch pixel normalization (`norm_pix_loss`).
- `forward(imgs, mask_ratio=0.75)` — returns `(loss, pred, mask)`.

**Architecture factory functions** (registered in `models_mae.__dict__` for lookup by `--model`):
`mae_vit_small/base/large/huge_patch16(...)` plus the SFM-specific
`mae_vit_base_patch16D4d256` and `mae_vit_large_patch16D4d256` (decoder embed 256, depth 4 — the
lightweight decoders actually used in `train.sh`/`submit-train.sh`).

### `SFM-Pretrain/main_pretrain.py` — pre-training entrypoint

- `get_args_parser()` — MAE-style args: `--batch_size --epochs --accum_iter --model
  --input_size(224) --mask_ratio(0.75) --blr --weight_decay --warmup_epochs --data_path
  --output_dir` plus distributed args. Also hard-codes `os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'`.
- `main(args)` execution order:
  1. `misc.init_distributed_mode` → device/seed/`cudnn.benchmark`.
  2. `dataset_train = SeismicSet(args.data_path, args.input_size)` (see datasets below).
  3. `DistributedSampler` + `DataLoader` (drop_last).
  4. `model = models_mae.__dict__[args.model](norm_pix_loss=..., in_chans=1)`.
  5. Compute LR by linear scaling rule: `lr = blr * eff_batch_size / 256` where
     `eff_batch_size = batch_size * accum_iter * world_size`.
  6. Wrap in `DistributedDataParallel` (find_unused_parameters=True).
  7. `AdamW` (betas 0.9/0.95) with timm `add_weight_decay` (no WD on bias/norm) + `NativeScaler`.
  8. Loop epochs → `engine_pretrain.train_one_epoch` → save checkpoint every 2 epochs → append
     `log.txt` and TensorBoard scalars.

### `SFM-Pretrain/engine_pretrain.py`

- `train_one_epoch(...)` — standard MAE loop: per-iteration cosine LR via
  `lr_sched.adjust_learning_rate`, `model(samples, mask_ratio=args.mask_ratio)` → loss,
  gradient accumulation via `accum_iter`, `loss_scaler` (AMP grad scaling — note autocast is
  actually disabled: `torch.cuda.amp.autocast(enabled=False)`), TensorBoard logging with an
  `epoch_1000x` x-axis. Ignores labels (`(samples, _)`), consistent with self-supervision.

### `SFM-Finetune/util/datasets.py` — all Dataset classes

This is the data contract for every task. All datasets read raw `float32` `.dat` files.

- `SeismicSet` (pre-train) — lists files under `data_path`, each read as
  `(1, input_size, input_size)` and **z-score normalized** (`(d-mean)/std`); returns
  `(tensor, dummy_label)`.
- `FacesSet` (SEAM / facies) — expects `<folder>seismic/<i>.dat` and `<folder>label/<i>.dat`
  for `i in 0..116`, shape 768×768. Train = first 100, val = last 17. Label is `int` and
  **shifted by -1** (`label-1`) → 6 classes.
- `SaltSet` (Salt/geobody) — `<folder>seismic/<i>.dat` + `label`, `i in 0..3999`, 224×224.
  Train 3500 / val 500. Labels are int (2 classes).
- `DenoiseSet` — train: `seismic/<i>.dat` (noisy) + `label/<i>.dat` (clean), `i in 0..1999`.
  val: `field/<i>.dat` for `i in 0..3999` (field data, label = itself).
- `ReflectSet` (Reflection/inversion) — train: `seismic/` + `label/`, `i in 0..2199`.
  val: `SEAMseismic/` + `SEAMreflect/`, `i in 0..3999`. Both seismic and label are z-score
  normalized.
- `InterpolationSet` — train: `<folder><i>.dat` for `i in 0..5999`; val: `<folder>U<i>.dat`.
  Returns `(d, d)` (input == target; masking is applied inside the model).

> Gotcha: file counts/paths are **hard-coded index ranges**, not globbing. Your data must match
> the exact naming (`seismic/0.dat`, `label/0.dat`, `U2000.dat`, ...) or `np.fromfile` returns
> empty and `reshape` fails.

### `SFM-Finetune/main_finetune.py` — fine-tuning entrypoint

- `get_args_parser()` — adds `--task {SEAM|Salt|Denoise|Reflection|Interpolation}`,
  `--finetune <ckpt>` (load pre-trained MAE weights), `--frozen` (freeze backbone, train only
  decoder + segmentation head), `--modelComparsion {Deeplab|Unet}` (train a baseline instead),
  `--nb_classes`, `--input_size(768)`, layer-wise LR decay, drop-path, random-erase, etc.
- `main(args)` execution order:
  1. Select dataset + `nb_classes` by `args.task` (dispatch block).
  2. Build model:
     - `--modelComparsion Deeplab` → `DeepLab(...)` (bundled baseline);
     - `--modelComparsion Unet` → `U_Net(...)`;
     - else task in `{Denoise, Interpolation, Reflection}` → `models_Regression.__dict__[args.model](...)`;
     - else task in `{SEAM, Salt}` → `models_Segmentation.__dict__[args.model](...)`.
  3. If `--finetune` and not `--eval`: `torch.load` checkpoint, drop mismatched `head.*` keys,
     `interpolate_pos_embed` (resize positional embeddings if input size differs from pre-train),
     `load_state_dict(strict=False)`.
  4. If `--frozen`: freeze all params, then unfreeze `model.decoder` and `model.segmentation_head`.
  5. `AdamW` over trainable params + `NativeScaler`.
  6. Criterion: regression tasks → `models_Regression.forward_loss` (0.5·MSE + 0.5·MS-SSIM);
     segmentation tasks → `CrossEntropyLoss`.
  7. Train loop → `engine_finetune.train_one_epoch`; evaluate with `evaluate` (acc + mIoU) for
     SEAM/Salt or `evaluateRegression` (MSE, MS-SSIM, PSNR) for the regression tasks.
     `--eval` runs evaluation only and exits.

### `SFM-Finetune/models_Segmentation.py` — classification/segmentation backbone

- `class VisionTransformer(timm...VisionTransformer)` — subclasses timm's ViT and adds a
  `VIT_MLAHead` `decoder` + a `SegmentationHead`. `forward_features` collects **4 intermediate
  block outputs** (every `depth//4` blocks) as multi-level features and feeds them to the MLA
  head, which fuses and upsamples them (×16 total) back to a full-resolution class map.
- Helper modules: `Conv2dReLU`, `DecoderBlock`, `SegmentationHead`, `DecoderCup`, `MLAHead`,
  `VIT_MLAHead` (Multi-Level Aggregation head, SETR-style).
- Factory funcs: `mae_vit_small_patch16`, `vit_base_patch16` (embed 768/depth 12),
  `vit_large_patch16` (embed 1024/depth 24), `vit_huge_patch14`. Selected via `--model`.

### `SFM-Finetune/models_Regression.py` — regression backbone

- `class VisionTransformer(...)` with `Interpolation` flag. For interpolation it calls
  `generate_mask(x, 0.75)` to zero out 16-px-wide vertical stripes (simulating missing traces),
  reconstructs, and computes loss only on the masked region.
- `forward_Interpolationloss` = `0.9·MSE + 0.1·MS-SSIM` on the reconstructed masked pixels.
- Module-level `forward_loss(imgs, pred)` = `0.5·MSE + 0.5·MS-SSIM` (used for Denoise/Reflection).
- Same `vit_base/large/huge` factory functions as the segmentation file.

### `SFM-Finetune/engine_finetune.py`

- `train_one_epoch(model, criterion, ...)` — supervised loop with layer-wise LR decay,
  gradient accumulation, TensorBoard logging.
- `evaluate(...)` — segmentation metrics via `util.tools.accuracy` → pixel **acc** and **mIoU**.
- `evaluateRegression(..., task=...)` — computes **MSE**, **MS-SSIM** (`util.msssim.MSSSIM`) and
  **PSNR** (`util.msssim.PSNR`).

### Bundled baselines (`SFM-Finetune/modules/modeling/`)

Vendored copies of **DeepLabV3+** (with ResNet/Xception/DRN/MobileNet backbones and
`sync_batchnorm`) and a **Nested U-Net**. These are only instantiated when `--modelComparsion`
is set, so you can benchmark SFM against conventional task-specific CNNs under identical
data/hyperparameters.

## Execution Flow

**Pre-training (self-supervised):**
```
train.sh / submit-train.sh
   └─> main_pretrain.py
         ├─ SeismicSet(data_path)                      # 224×224 float32 patches, z-scored
         ├─ models_mae.mae_vit_base_patch16D4d256(in_chans=1)
         ├─ AdamW + NativeScaler, lr = blr*eff_bs/256
         └─ for epoch: engine_pretrain.train_one_epoch # mask 75% -> reconstruct -> MSE on masked
               └─ save checkpoint-*.pth every 2 epochs to --output_dir
```

**Fine-tuning (supervised, per task):**
```
finetune-*.sh
   └─> main_finetune.py --task T --finetune <pretrained.pth>
         ├─ Dataset for T (FacesSet/SaltSet/DenoiseSet/ReflectSet/InterpolationSet)
         ├─ build model: models_Segmentation (SEAM/Salt) or models_Regression (Denoise/Reflection/Interpolation)
         │     └─ load pretrained MAE encoder weights (strict=False) + interpolate_pos_embed
         ├─ criterion: CrossEntropy (seg) or 0.5*MSE+0.5*MS-SSIM (reg)
         └─ for epoch: engine_finetune.train_one_epoch
               └─ evaluate (acc/mIoU) or evaluateRegression (MSE/MS-SSIM/PSNR)
                     └─ save checkpoint + log.txt + TensorBoard
```

**Visualization:** `SFM-Finetune/Application/visualization.ipynb` loads a fine-tuned checkpoint
(e.g. `facies.pth`) and renders predictions vs. ground truth.

## How To Run The Repository

> ⚠️ **Environment mismatch with Boglodite.** This repo pins **PyTorch 1.8.1 + CUDA 11.1** and
> **`timm==0.3.2`** (with a hard version `assert` in `main_pretrain.py`). timm 0.3.2 requires the
> well-known one-line patch (`from torch._six import container_abcs` →
> `import collections.abc as container_abcs`) to import on modern PyTorch. GPU + the external
> `.dat` datasets/checkpoints (tens of GB) are required for any real run — none ship with the repo.

### 1. Environment (documented in top `README.md` — unverified here)

```shell
conda create -n SFM python=3.9.12 && conda activate SFM
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
  -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt          # timm==0.3.2 scikit-learn matplotlib tensorboard
# optional visualization:
pip install jupyter notebook && python -m ipykernel install --user --name=SFM
```

**Boglodite note:** per repo conventions use `uv add` instead of `pip install` (e.g.
`uv add "timm==0.3.2" scikit-learn matplotlib tensorboard`) and run scripts with
`uv run python ...`. Be aware the pinned Torch 1.8.1+cu111 wheels are old and may not resolve on
current CUDA — a newer Torch + the timm patch is usually needed. Stream long GPU runs through
`tee -a outputs/run.log`.

### 2. Get data & pre-trained weights (external downloads)

- Pre-train data + all downstream task data: links in `Data/README-*.md`.
- Pre-trained checkpoints (SFM-Base/Large, 224/512): "Model Zoo" table in top `README.md`.
- Multi-part pre-train zip must be merged on Linux first:
  `zip -s 0 mae_data_more.zip --out pretrain.zip && unzip pretrain.zip`.
- Place downstream data under `Data/<Task>/` and checkpoints under `SFM-Pretrain/output_dir/`.
- In Boglodite, put datasets under `data/` and reusable weights under `models/`, then point the
  `--data_path` / `--finetune` flags at those locations.

### 3. Pre-train (multi-GPU; documented)

```shell
cd SFM-Pretrain
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --batch_size 580 --accum_iter 4 \
    --model mae_vit_base_patch16D4d256 --mask_ratio 0.75 \
    --epochs 1600 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ../Data/Pretrain/mae_data_more/ \
    --output_dir ./output_model/ --log_dir ./output_model/
```

### 4. Fine-tune a downstream task (single GPU; documented)

Facies (classification, uses the `finetune-Facies.sh` pattern):
```shell
cd SFM-Finetune
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py \
    --data_path ../Data/Facies/ --task SEAM \
    --model vit_base_patch16 --input_size 768 --nb_classes 6 \
    --finetune ../SFM-Pretrain/output_dir/Base-512.pth \
    --accum_iter 2 --batch_size 1 --epochs 100 --warmup_epochs 10 \
    --blr 1.5e-3 --weight_decay 0.05 --layer_decay 0.05 --drop_path 0.1 --reprob 0.25 --dist_eval \
    --output_dir ./Application/Facies/modelbase_512/ --log_dir ./Application/Facies/modelbase_512/
```

Other tasks — set `--task` and `--input_size 224` and point `--data_path` accordingly:
`Salt` (`../Data/Geobody/`), `Denoise` (`../Data/Denoise/`), `Reflection` (`../Data/Inversion/`),
`Interpolation` (`../Data/Interpolation/`). Ready-made examples live in
`SFM-Finetune/Application/finetune-{Salt,Denoise,Reflect,Interpolation}.sh`.

Baseline comparison: add `--modelComparsion Unet` or `--modelComparsion Deeplab`.
Frozen-backbone (linear-probe-style) transfer: add `--frozen 1 --finetune <ckpt>`.

### 5. Evaluate / visualize

- Add `--eval` to `main_finetune.py` for evaluation-only on a `--resume`d checkpoint
  (segmentation reports acc/mIoU).
- Open `SFM-Finetune/Application/visualization.ipynb` for qualitative prediction plots.

## Important Constraints Or Gaps

- **No data, no weights in-repo** — everything substantive is an external USTC/Baidu download.
  The repo is code-only + README stubs.
- **Hard-coded dataset indexing** in `util/datasets.py` (fixed file-count ranges and folder names
  like `seismic/`, `label/`, `field/`, `SEAMseismic/`, `U<n>.dat`). New data must match exactly,
  or write a new `Dataset` class and register it in `main_finetune.py` (README documents this
  5-step recipe).
- **Version fragility:** `assert timm.__version__ == "0.3.2"` (pretrain) and the timm/PyTorch 1.8
  compatibility patch. The finetune entrypoint comments out its own version assert.
- **Single-channel seismic** (`in_chans=1`); `unpatchify` in `models_mae.py` is hard-coded to 3
  channels (dead code on the loss path but a latent bug if reused for reconstruction viz).
- **Minor script typo:** `Application/finetune-Salt.sh` has an unbalanced quote in
  `--data_path ../Data/Geobody'` — fix before running.
- **AMP autocast disabled** in `engine_pretrain.py` (`autocast(enabled=False)`) despite using a
  grad scaler — training runs in fp32.
- **Licensing:** downstream sub-dirs are **CC-BY-NC 4.0** (non-commercial), even though the top
  README says MIT. Ported/adapted MAE + DeepLab + U-Net code carries upstream licenses too.
- **No test suite / no packaging** (`pyproject.toml`/`setup.py` absent) — it is a research
  training codebase driven entirely by the `*.sh` launch scripts.

## Bottom Line

SFM is a **research-grade MAE-based seismic foundation model**: a ViT encoder self-supervised on
millions of 224×224 seismic patches, then fine-tuned via task-specific decoder heads for facies
classification, geobody/salt segmentation, denoising, reflectivity inversion, and interpolation,
with bundled U-Net/DeepLab baselines for comparison. Everything is driven by `main_pretrain.py`
and `main_finetune.py` through the `SFM-Pretrain/*.sh` and `SFM-Finetune/Application/*.sh`
scripts. To actually run it you must supply a GPU, the external `.dat` datasets, and (for
transfer) the external pre-trained checkpoints, and reconcile the old PyTorch 1.8.1 / `timm==0.3.2`
pins with the current environment (using `uv add` / `uv run` per Boglodite conventions).
