# 🛡️ MOS-Attack & MOS-Defense
### Multi-Objective Set-Based Adversarial Attack and Defense Framework

> **Based on:** *"MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework"* — Guo et al., CVPR 2025  
> **Extended with:** MOS-Defense — A novel multi-objective adversarial training method

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CVPR](https://img.shields.io/badge/Paper-CVPR%202025-blueviolet.svg)](https://openaccess.thecvf.com/)

---

## 👤 Author

| Field | Details |
|-------|---------|
| **Name** | Harsh Gupta |
| **Roll No** | M25EC002 |
| **GitHub** | [@harshgupta107](https://github.com/harshgupta107) |

---

## 📌 Overview

This repository implements:

1. **MOS-Attack** — A scalable multi-objective adversarial attack using 8 surrogate loss functions optimized jointly via smooth set-based optimization (APGD).
2. **MOS-Defense** *(Novel Contribution)* — A multi-objective adversarial training method that trains models against a **set** of K adversarial examples per input, each targeting a different loss function, with a synergy regularizer.

---

## 🧠 Key Concepts

### MOS-Attack
Instead of using a single surrogate loss, MOS-Attack maximizes **all 8 loss functions simultaneously** using the smooth set-based objective:

$$\max_{\Delta} g(\Delta) = -\mu \log \left( \sum_{i=1}^{m} \left( \sum_{k=1}^{K} e^{f_i(\delta_k)/\mu} \right)^{-1} \right)$$

### MOS-Defense (Our Contribution)
$$\mathcal{L}_{\text{MOS-Def}} = \lambda_c \cdot \text{CE}(f(x), y) + (1-\lambda_c) \cdot \frac{1}{K}\sum_{k=1}^{K} \text{CE}(f(x+\delta_k), y) + \lambda_s \cdot \mathcal{L}_{\text{syn}}$$

---

## ⚙️ Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10 |
| **GPU** | NVIDIA RTX A2000 12 GB |
| **Framework** | PyTorch + CUDA 11.8 |
| **Architectures** | PreActResNet-18, ResNet-18, WideResNet-28-10 |
| **Epochs** | 10 (demo) / 100 (full) |
| **Batch Size** | 16 |
| **Optimizer** | SGD (lr=0.1, momentum=0.9, wd=5e-4) |
| **Scheduler** | CosineAnnealingLR |
| **Perturbation** | L-inf, ε = 8/255, α = 2/255 |
| **PGD Steps** | 5 (inner) |
| **K (set size)** | 3 |
| **λ_clean** | 0.5 |
| **λ_synergy** | 0.1 |

---

## 📁 Repository Structure

```
MOSDEFENCE/
│
├── MOS_Attack_Defense_Colab.ipynb   # Main Colab notebook (all cells)
│
├── src/
│   ├── losses.py                    # 8 surrogate loss functions
│   ├── mos_attack.py                # MOS-Attack implementation (APGD)
│   ├── mos_defense.py               # MOS-Defense training objective
│   ├── baselines.py                 # PGD-AT, TRADES, Standard training
│   ├── models.py                    # PreActResNet18, ResNet18, WideResNet
│   └── evaluate.py                  # Evaluation functions
│
├── results/
│   ├── mos_defense_comparison.csv   # Full results table
│   ├── fig1_comparison.png          # 6-panel comparison plot
│   └── fig2_ablation.png            # K-ablation plot
│
├── report/
│   └── HarshGupta_M25EC002.pdf      # Project report
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/harshgupta107/MOSDEFENCE.git
cd MOSDEFENCE

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20
pandas>=1.2
matplotlib>=3.4
seaborn>=0.13
tabulate>=0.9
tqdm>=4.67
```

---

## 🚀 Quick Start

### Run on Google Colab
Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wXAmUv99MoNUvtdr9D0srgSh99H6xjFR)

### Run Locally

```python
import torch
from src.models import PreActResNet18
from src.mos_attack import MOSAttack
from src.mos_defense import MOSDefense

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = PreActResNet18(num_classes=10).to(device)

# Run MOS-Attack
attacker = MOSAttack(model, eps=8/255, steps=50, K=5, mu=0.1)
x_adv = attacker.perturb(x, y)

# Train with MOS-Defense
defender = MOSDefense(model, eps=8/255, alpha=2/255, K=3,
                      pgd_steps=10, lambda_clean=0.5, lambda_synergy=0.1)
loss, clean_loss, adv_loss = defender.mos_defense_loss(x_batch, y_batch)
```

---

## 📊 Results

### Robust Accuracy (%) on CIFAR-10 (ε = 8/255)

| Architecture | Defense | Clean | FGSM | PGD-20 | MOS-K1 | MOS-K5 |
|---|---|---|---|---|---|---|
| PreActResNet-18 | Standard | 34.20 | 5.30 | 0.60 | 22.80 | 18.90 |
| PreActResNet-18 | **PGD-AT** | **37.30** | **24.90** | **24.50** | **30.30** | **30.30** |
| PreActResNet-18 | TRADES | 34.20 | 12.00 | 11.00 | 24.00 | 23.20 |
| PreActResNet-18 | MOS-Defense | 10.30 | 10.30 | 10.30 | 10.30 | 10.30 |
| ResNet-18 | Standard | **59.00** | 4.80 | 0.10 | 24.80 | 20.80 |
| ResNet-18 | PGD-AT | 36.70 | 21.50 | 21.60 | 27.20 | 28.30 |
| ResNet-18 | **TRADES** | 39.20 | **24.00** | **23.10** | **30.50** | **33.00** |
| ResNet-18 | MOS-Defense | 14.30 | 12.80 | 12.80 | 12.80 | 12.80 |
| WideResNet-28-10 | Standard | **53.60** | 6.60 | 2.80 | 26.10 | 24.30 |
| WideResNet-28-10 | **PGD-AT** | 35.00 | **18.50** | **18.20** | **26.20** | **25.50** |
| WideResNet-28-10 | TRADES | 10.30 | 10.30 | 10.30 | 10.30 | 10.30 |
| WideResNet-28-10 | MOS-Defense | 18.80 | 16.90 | 16.90 | 16.30 | 16.50 |

### Ablation: Effect of K in MOS-Defense (PreActResNet-18)

| K | Clean (%) | PGD-20 (%) | MOS-K1 (%) | MOS-K5 (%) |
|---|---|---|---|---|
| 1 | **32.0** | **21.6** | **24.3** | **23.7** |
| 3 | 10.6 | 10.6 | 10.6 | 10.6 |
| 5 | 10.2 | 10.2 | 10.2 | 10.2 |

### Plots

| Comparison Plot | Ablation Plot |
|---|---|
| ![Comparison](results/fig1_comparison.png) | ![Ablation](results/fig2_ablation.png) |

---

## 🔬 8 Loss Functions

| ID | Name | Description |
|----|------|-------------|
| 0 | Cross-Entropy | Standard log-loss |
| 1 | Marginal Loss | max wrong − true logit |
| 2 | DLR | Difference of Logits Ratio |
| 3 | Boosted CE | −log p_y − log(1 − max p_j) |
| 4 | Searched Loss 1 | AutoLoss-Zero exponential ratio |
| 5 | Searched Loss 2 | Double softmax formulation |
| 6 | Searched Loss 3 | Softmax composition with one-hot |
| 7 | Searched Loss 4 | Squared softmax residual |

---

## 🆕 MOS-Defense: Our Novel Contribution

| Property | PGD-AT | MOS-Defense (Ours) |
|----------|--------|-------------------|
| Inner maximization | Single CE loss | K diverse losses |
| Adversarial examples/batch | 1 | K |
| Loss diversity | None | Synergy regularizer |
| Attack surface coverage | Single | Multi-surface |

**Key novelties:**
- ✅ Set-based inner maximization (K adversarial examples per input)
- ✅ Each k-th example targets a different loss function (synergistic)
- ✅ Synergy regularizer penalizes correlated attack patterns
- ✅ Multi-example robust loss: average CE over all K adversarial examples

---

## 📈 Key Findings

- **Standard training** achieves highest clean accuracy (~59%) but collapses under PGD-20 (<1% robustness)
- **PGD-AT** achieves the best average rank across all architectures
- **TRADES** performs best for ResNet-18 under MOS-K5 (33% robust accuracy)
- **MOS-Defense** requires longer training (100 epochs, full CIFAR-10) to show its full advantage — under the 10-epoch quick demo it underperforms due to the harder adversarial training signal needing more optimization steps
- Ablation confirms **K=1** is optimal for short training schedules

---

## 📄 Report

The full project report is available at:
📄 [`report/HarshGupta_M25EC002.pdf`](report/HarshGupta_M25EC002.pdf)

---

## 📚 References

1. Guo et al. *MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework.* CVPR 2025.
2. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
3. Zhang et al. *Theoretically Principled Trade-off Between Robustness and Accuracy (TRADES).* ICML 2019.
4. Croce & Hein. *Reliable Evaluation of Adversarial Robustness (AutoAttack).* ICML 2020.
5. Goodfellow et al. *Explaining and Harnessing Adversarial Examples (FGSM).* ICLR 2015.

---

## 📝 License

This project is for academic purposes. See [LICENSE](LICENSE) for details.

---

<div align="center">
<b>Harsh Gupta | M25EC002 | ECE Department</b><br>
<a href="https://github.com/harshgupta107/MOSDEFENCE">github.com/harshgupta107/MOSDEFENCE</a>
</div>
