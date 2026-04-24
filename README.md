<div align="center">

# 🧩 Dr.LLM: Dynamic Layer Routing in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2510.12773-b31b1b.svg)](https://arxiv.org/pdf/2510.12773)</br>
<a href="https://www.linkedin.com/in/ahmed-heakl/"><b>Ahmed Heakl</b></a>, <a href="https://scholar.google.com/citations?user=Jt4OYwMAAAAJ&hl=fr"><b>Martin Gubri</b></a>, <a href="https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en"><b>Salman Khan</b></a>, <a href="https://scholar.google.com/citations?user=o0qtjzYAAAAJ&hl=en"><b>Sangdoo Yun</b></a>, <a href="https://seongjoonoh.com/"><b>Seong Joon Oh</b></a></br>
Parameter Lab · MBZUAI · NAVER AI Lab · University of Tübingen · Tübingen AI Center


---

### 🚨 Code Release Status
🧩 The **training**, **data generation**, and **in-domain evaluation** code for **Dr.LLM** are **not yet released**.  
These components (MCTS supervision, router training scripts, and lm-eval integration) will be made public in an upcoming update.  
**Stay tuned for the full release!**

</div>


## 🆕 Latest Updates
- 🚀 **24 April 2026**: Data generation code released!
- 🏆 **3 February 2026**: Paper accepted at [TTT @ ICLR 2026](https://iclr.cc/virtual/2026/workshop/10000776)!
- 🎉 **25 January 2026**: Paper accepted at [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011611)!
- 📢 **15 October 2025**: Paper ArXived!

## 📘 Table of Contents
- [Overview](#overview)
- [🧪 Evaluation](#-evaluation)
  - [In-Domain (Training & Evaluation Tasks)](#in-domain-training--evaluation-tasks)
  - [Out-of-Domain (Generalization Benchmarks)](#out-of-domain-generalization-benchmarks)
- [📊 Results Summary](#-results-summary)
- [⚙️ Usage](#️-usage)
  - [Installation](#1️⃣-installation)
  - [Data Generation with MCTS](#2️⃣-data-generation-with-mcts)
  - [Training the Routers](#2️⃣-training-the-routers)
  - [Evaluation with lm-eval-harness](#3️⃣-evaluation-with-lm-eval-harness)
- [🧭 Citation](#-citation)


## 🧩 Overview

<p align="center">
  <img src="assets/teaser.png" width="50%" alt="Dr.LLM Teaser">
</p>

Large Language Models (LLMs) process every token through all layers of a transformer stack, wasting compute on simple queries and lacking flexibility for harder ones that need deeper reasoning.  

**Dr.LLM (Dynamic Routing of Layers for LLMs)** is a retrofittable framework that adds lightweight per-layer routers to pretrained models.  
Each router decides whether to skip, execute, or repeat a layer, enabling adaptive depth without retraining or architectural changes.

Routers are trained with explicit supervision from Monte Carlo Tree Search (MCTS), generating high-quality layer configurations that preserve or improve accuracy under a compute budget.  
Stabilized with windowed pooling, focal loss, and bottleneck MLPs, Dr.LLM maintains robustness under class imbalance and long sequences.

📈 **Results**
- On ARC (logic) and DART (math), Dr.LLM improves accuracy by **+3.4%p** while saving **~5 layers** per input.
- Routers generalize to MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, and AGIEval with only **0.85% accuracy drop**.
- Outperforms prior routing methods (LayerSkip, FlexiDepth, MindSkip) by up to **+7.7%p**.

> 💡 Dr.LLM equips frozen LLMs for **budget-aware**, **accuracy-driven inference** — no base weight modification required.

### Routers
<p align="center">
  <img src="assets/routers_architecture.png" width="80%" alt="Dr.LLM Teaser">
</p>

> Our layer routing based on hidden states. Dr.LLM augments a frozen decoder-only LLM with per-layer routers that decide to skip, execute, or repeat a block once. Routers read windowed summaries of hidden states
and are trained from MCTS-derived targets. 

### Training with MCTS Supervision
<p align="center">
  <img src="assets/training_mcts.png" width="95%" alt="Dr.LLM Teaser">
</p>

> Length-aware MCTS used to collect the supervised training dataset of per-layer routing
configurations (skip/execute/repeat). For each input, MCTS explores modified layer paths
and retains accuracy-preserving or improving ones under a compute budget.

## 🧪 Evaluation

We evaluate **Dr.LLM** using [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness) across **in-domain** and **out-of-domain** benchmarks.

### In-Domain (Training & Evaluation Tasks)
Routers are trained and evaluated on **ARC-Easy/Challenge** (logic) and **DART-Math (levels 1–5)** (multi-step math reasoning), using 4K MCTS-derived execution paths.

| Dataset | Domain | Metric |
| -------- | ------- | ------- |
| ARC-Easy / Challenge | Logic Reasoning | Accuracy |
| DART (levels 1–5) | Math Reasoning | Accuracy |

### Out-of-Domain (Generalization Benchmarks)
We test zero-shot transfer on **MMLU**, **GSM8k**, **AIME24**, **TruthfulQA**, **GPQA Diamond**, **AGIEval**, **SQuADv2**, and **PIQA**.  
All evaluations follow default `lm-eval-harness` settings (2048 max tokens, greedy decoding).



---

## ⚙️ Usage

### 1️⃣ Installation

```bash
git clone https://github.com/parameterlab/dr-llm
cd dr-llm
pip install -r requirements.txt
```

### 2️⃣ Data Generation with MCTS

The data generation pipeline uses **length-aware MCTS** to discover optimal per-layer routing configurations (skip/execute/repeat) for each training example.

#### Supported Models

Modified model files compatible with the data generation pipeline are provided in `data_models/`:

```
data_models/
├── modeling_llama.py
├── modeling_qwen2.py
├── modeling_qwen3.py
└── ...
```

These files expose a `layer_indices` attribute on the base model class, which the MCTS search manipulates at runtime to explore different execution paths — no weight modification required. See `data_models/README.md` for instructions on adapting a new model architecture.

#### Running Data Generation

```bash
python data_generation.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc,dart \
  --output_dir data/mcts_paths \
  --num_simulations 50 \
  --budget 2
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | — | HuggingFace model path or ID |
| `--dataset` | `arc,dart` | Comma-separated list of datasets |
| `--num_simulations` | `50` | MCTS simulations per example |
| `--budget` | `2` | Max path length factor (cap at `2L`) |
| `--output_dir` | `data/` | Where to save routing configurations |

#### Output Format

Each output file contains MCTS-derived tuples `(question, optimal_layer_config, answer)` where `optimal_layer_config` is a vector of `{0=skip, 1=execute, 2=repeat}` labels of length `L` (number of layers). These are used directly as supervision targets for router training.


<details>
<summary><b>2️⃣ Training the Routers </b></summary>

> ⚠️ Note: Full code release is pending ⚠️

Training uses **AdamW**, 25 epochs, **1×10⁻³ LR**, **bf16 precision**, and a **single A100 GPU (40GB)** — taking under 4 hours.


Models source code must be manipulated to insert routers after each transformer block. 

Routers are trained separately using MCTS-generated supervision:

```bash
python train.py \
  --model llama-3-8b-instruct \
  --data_dir data/arc_dart \
  --save_dir checkpoints/drllm_router
```
</details>

<details>
<summary><b>3️⃣ Evaluation with lm-eval-harness</b></summary>

> 🚨 Note: Full code release is pending 🚨

```bash
lm_eval \
  --model openai/llama-3-8b-instruct \
  --tasks arc_challenge,dart,gsm8k,mmlu \
  --device cuda
```
</details>


---


## 🧭 Citation

If you find this work useful, please cite:

```bibtex
@article{heakl2025drllm,
  title={Dr.LLM: Dynamic Layer Routing in LLMs},
  author={Ahmed Heakl and Martin Gubri and Salman Khan and Sangdoo Yun and Seong Joon Oh},
  journal={arXiv preprint arXiv:2510.12773},
  year={2025}
}
```

