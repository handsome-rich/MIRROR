<div align="center">

# 🪞 MIRROR

### Manifold Ideal Reference ReconstructOR

**A Reference-Comparison Framework for Generalizable AI-Generated Image Detection**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2602.02222)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/handsome-rich/MIRROR?style=social)](https://github.com/handsome-rich/MIRROR)

[🇨🇳 中文文档](README_zh.md) · [📄 Paper](http://arxiv.org/abs/2602.02222) · [📊 Results](#-results) · [🚀 Quick Start](#-quick-start) · [📚 Citation](#-citation)

<img src="fig/intro2_01.png" width="92%" alt="MIRROR teaser">

</div>

---

> *"Perception is a process of hypothesis testing."*
> &nbsp;&nbsp;&nbsp;&nbsp;Richard L. Gregory, 1980

**MIRROR** reframes AI-generated image (AIGI) detection as a **Reference-Comparison** process. A learnable, diversity-regularized **Memory Bank** captures feature regularities of *real* images. Sparse top-$k$ attention reconstructs an input-conditioned **Ideal Reference**, and the **comparison residual** and retrieval uncertainty, together with global image context, drive the final detection score.

This design brings two properties:

- 📈 **Backbone scalability.** Accuracy keeps climbing as DINOv3 scales from Base to Huge, while the evaluated NPR / UnivFD / DDA configurations show limited gains.
- 👁️ **Human-level comparison.** On the original Human-AIGI set covering **27 generators**, DINOv3-L reaches **89.6%** and DINOv3-H+ reaches **93.6%** fake-image accuracy, exceeding the lay-user and CV-expert means and approaching the AIGI-expert mean. DINOv3-L retains **89.5%** on the human-imperceptible subset.

---

## 📰 News

- **2026.10** &nbsp;Two-phase training code released; inference code updated.
- **2026.04** &nbsp;Inference code and DINOv3-H+ weights released.
- **2026.03** &nbsp;Paper released on arXiv.
- **Coming soon** &nbsp;The Human-AIGI Benchmark.

---

## ✨ Highlights

| | |
|---|---|
| 🔄 **New paradigm** | Reference-Comparison instead of artifact hunting |
| 🏆 **State of the art** | +2.1 percentage points on 6 standard benchmarks, +8.1 points across 7 in-the-wild benchmarks |
| 👁️ **Human-level comparison** | Human-AIGI Orig.: 89.6% (L), 93.6% (H+); Hard: 89.5% (L) |
| 📈 **Scales with backbone** | Sustained gains from DINOv3-Base to -Huge in the evaluated configurations |
| 🧠 **Reality memory bank** | $K$ prototypes with a row-normalized Gram diversity penalty to encourage low coherence |
| 🔬 **First-of-its-kind benchmark** | Psychophysically curated Human-AIGI set, 50 participants, 27 generators |

---

## 🧭 How It Works

MIRROR is a two-phase framework (see `fig/method.pdf`).

### Phase 1 · Encoding Reality Priors

A frozen DINOv3 encoder extracts feature tokens $F \in \mathbb{R}^{N \times D}$ from *real* images only. We train the memory bank $\mathbf{M} \in \mathbb{R}^{K \times D}$ and all sparse top-$k$ attention projections using mean-squared reconstruction error and a row-normalized Gram diversity penalty:

$$
\mathcal{L}_{\text{Phase1}} \;=\; \frac{\lVert F - \hat{F} \rVert_F^2}{ND} \;+\; \lambda \,\lVert \bar{\mathbf{M}}\bar{\mathbf{M}}^{\top} - \mathbf{I}_K \rVert_F
$$

Here $N$ is the number of feature tokens, $D$ is the feature dimension, and $\bar{\mathbf{M}}$ is the row-normalized memory bank. The first term learns real-feature reconstruction; the second encourages low coherence and reduces prototype redundancy in the overcomplete bank. The default configuration uses $K=4096$ prototypes and retrieves $k=128$ entries per attention head.

### Phase 2 · Reference-Comparison Detection

The memory bank **and attention projector stay frozen**, while encoder **LoRA adapters and the detection head** are trained with real/fake supervision. Each input is reconstructed through the same sparse attention to form its Ideal Reference $\hat{F}$. The head combines the token-averaged **comparison residual**, **reconstruction perplexity** learned from attention maximum and entropy, and the encoder's global **CLS token** to predict the fake-class probability.

---

## 📊 Results

Paired real/fake benchmarks report **Balanced Accuracy (B.Acc., %)**; generated-only CO-SPY and Human-AIGI report **Fake Accuracy (F.Acc., %)**. Results use format-aligned inputs.

### 13 standard + in-the-wild benchmarks

| Category | Benchmark | Prior SOTA | DINOv2-L | DINOv3-L | **DINOv3-H+** | Δ vs SOTA |
|---|---|---:|---:|---:|---:|---:|
| **Standard** | AIGCDetectBenchmark | 84.7 (B-Free) | 90.5 | 91.7 | **97.3** | **+12.6** |
| | GenImage | 89.6 (B-Free) | 91.3 | 94.2 | **99.8** | **+10.2** |
| | UnivFakeDetect | 87.8 (B-Free) | 84.6 | 88.2 | **92.4** | **+4.6** |
| | Synthbuster | 96.5 (DDA) | 97.0 | 98.1 | **99.2** | **+2.7** |
| | EvalGEN | 96.6 (DDA) | 98.1 | 99.0 | **99.8** | **+3.2** |
| | DRCT-2M | 99.2 (B-Free) | 92.8 | 93.0 | 93.0 | −6.2 |
| **In-the-wild** | Chameleon | 83.5 (DDA) | 85.4 | 90.7 | **94.6** | **+11.1** |
| | SynthWildx | 94.6 (B-Free) | 88.9 | 93.1 | **95.1** | **+0.5** |
| | WildRF | 92.6 (B-Free) | 92.2 | 96.7 | **97.8** | **+5.2** |
| | AIGIBench | 84.4 (DDA) | 85.6 | 90.5 | **94.9** | **+10.5** |
| | CO-SPY | 80.3 (DDA) | 87.4 | 91.3 | **97.4** | **+17.1** |
| | RR-Dataset | 70.3 (DDA) | 76.8 | 78.9 | **88.3** | **+18.0** |
| | BFree-Online | 87.1 (B-Free) | 84.3 | 83.0 | **97.6** | **+10.5** |

> DINOv3-L improves the six-standard-benchmark average by **2.1 percentage points** over B-Free and the in-the-wild score by **8.1 points** over DDA. The in-the-wild score is the unweighted mean of **16 evaluation columns** across seven datasets.

### Human-AIGI · the 14th benchmark

A psychophysically curated benchmark covering **27 generators**, studied with **50 participants**. The Hard subset contains generated images whose mean perceived-realism rating is at least **3** on a 4-point scale, or whose mean response time exceeds $\mu_{rt}+\sigma_{rt}$. Here $\mu_{rt}$ and $\sigma_{rt}$ are the mean and standard deviation of per-image mean response times on Orig. Both Orig. and Hard contain generated images and use **F.Acc.**; the human reference levels below are measured on Orig.

| Method | Orig. F.Acc. (%) | Hard F.Acc. (%) |
|---|---:|---:|
| Lay users (untrained) | 76.9 | — |
| CV experts (no forensics training) | 88.3 | — |
| AIGI-detection experts | 94.5 | — |
| **MIRROR (DINOv3-L)** | **89.6** | **89.5** |
| **MIRROR (DINOv3-H+)** | **93.6** | — |

See the paper for full psychophysics and the per-generator breakdown.

---

## 🛣️ Roadmap

- [x] Inference code
- [x] DINOv3-H+ inference weights
- [x] Training code (Phase 1 + Phase 2)
- [ ] Human-AIGI Benchmark public release

---

## 🚀 Quick Start

### 1. Environment

Python 3.10+ is recommended.

```bash
git clone https://github.com/handsome-rich/MIRROR.git
cd MIRROR

# Install PyTorch first per your CUDA version: https://pytorch.org
pip install -r requirements.txt
```

### 2. Download Weights

| File | Purpose | Link |
|---|---|---|
| `checkpoint-h-cur.pth` | Phase 2 detector checkpoint | [Google Drive](https://drive.google.com/file/d/1gos1QgZA4Xuj706oa5i5E6vsOAoaLyr3/view?usp=sharing) |
| `mirror_phase1.pth` | Phase 1 memory-bank weights | [Google Drive](https://drive.google.com/file/d/1CpgltI-F7JN7hDyk2O16Ix3Zr_2d2-G0/view?usp=sharing) |
| `dinov3-huge/` | DINOv3-H+ backbone | official [DINOv3 release](https://github.com/facebookresearch/dinov3) |

Place them under `weight/`:

```text
weight/
├── checkpoint-h-cur.pth        # Phase 2 detector
├── mirror_phase1.pth           # Phase 1 memory bank
└── dinov3-huge/                # DINOv3-Huge backbone
    ├── config.json
    └── model.safetensors
```

### 3. Run Inference

```bash
python inference.py \
  --model_path     ./weight/checkpoint-h-cur.pth \
  --memory_path    ./weight/mirror_phase1.pth \
  --backbone_path  ./weight/dinov3-huge \
  --base_data_path /path/to/your/dataset \
  --benchmarks     Chameleon \
  --batch_size 128 \
  --device cuda \
  --use_amp
```

### 4. Dataset Layout

`--base_data_path` should point at a root that holds one folder per benchmark:

```text
base_data_path/
├── AIGC_bm/                # AIGCDetectBenchmark
├── UniversalFakeDetect/    # UnivFD
├── synthbuster/            # Synthbuster
├── GenEval-JPEG/           # EvalGEN
├── Chameleon/test/
├── WildRF/test/
├── synthwildx/
├── AIGIBench/
├── CO-SPY-In-the-Wild/
├── drct/
├── RRDataset/
└── B-Free/
```

For training, place real and AI-generated images in `datasets/train/0_real/` and `datasets/train/1_fake/`; use the same structure under `datasets/val/` for validation.

### 5. Train MIRROR

Example commands for two-phase training:

**Phase 1 · Memory Bank**

```bash
python mmbank.py \
  --dino_path  ./weight/dinov3-huge \
  --input_path ./datasets/train/0_real \
  --save_path  ./weight/phase1 \
  --epochs 100
```

**Phase 2 · Detector**

```bash
python main_finetune.py \
  --backbone_path  ./weight/dinov3-huge \
  --memory_path    ./weight/phase1/mirror_phase1_epoch_100.pth \
  --data_path      ./datasets/train \
  --eval_data_path ./datasets/val \
  --output_dir    ./weight/phase2 \
  --device cuda \
  --num_workers 8 \
  --epochs 100
```

---

## ⚙️ Inference Arguments

| Flag | Type | Description |
|---|---|---|
| `--model_path` | str | Phase 2 checkpoint (`.pth`) |
| `--memory_path` | str | Phase 1 memory-bank weights; optional when the Phase 2 checkpoint includes the memory bank |
| `--backbone_path` | str | DINOv3 backbone directory |
| `--base_data_path` | str | Root directory containing benchmark sub-folders |
| `--benchmarks` | list | Benchmarks to evaluate, e.g. `Chameleon GenImage` |
| `--batch_size` | int | Per-device batch size |
| `--device` | str | `cuda` or `cpu` |
| `--use_amp` | flag | Enable mixed-precision inference |
| `--output_dir` | str | Where CSV reports go (default `./results`) |

CSV reports land at `results/{benchmark}_{timestamp}.csv` with `Acc`, `Bal_Acc`, `Real_Acc`, `Fake_Acc`.

---

## 📚 Citation

If MIRROR helps your research, please cite:

```bibtex
@article{liu2026mirror,
  title={MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection},
  author={Liu, Ruiqi and Cui, Manni and Qin, Ziheng and Yan, Zhiyuan and Chen, Ruoxin and Han, Yi and Li, Zhiheng and Chen, Junkai and Chen, ZhiJin and Lin, Kaiqing and others},
  journal={NeurIPS 2026},
  year={2026}
}
```

---

## 📬 Contact

- **Issues** &nbsp;[github.com/handsome-rich/MIRROR/issues](https://github.com/handsome-rich/MIRROR/issues)
- **Email** &nbsp;`ruiqi.liu24@nlpr.ia.ac.cn`

---

<div align="center">
<sub>🪞 Built on the conviction that <em>understanding the real</em> generalizes further than <em>chasing the fake</em>.</sub>
</div>
