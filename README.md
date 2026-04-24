<div align="center">

# 🧩 Dr.LLM: Dynamic Layer Routing in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2510.12773-b31b1b.svg)](https://arxiv.org/pdf/2510.12773)</br>
<a href="https://www.linkedin.com/in/ahmed-heakl/"><b>Ahmed Heakl</b></a>, <a href="https://scholar.google.com/citations?user=Jt4OYwMAAAAJ&hl=fr"><b>Martin Gubri</b></a>, <a href="https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en"><b>Salman Khan</b></a>, <a href="https://scholar.google.com/citations?user=o0qtjzYAAAAJ&hl=en"><b>Sangdoo Yun</b></a>, <a href="https://seongjoonoh.com/"><b>Seong Joon Oh</b></a></br>
Parameter Lab · MBZUAI · NAVER AI Lab · University of Tübingen · Tübingen AI Center

</div>


## 🆕 Latest Updates
- 🚀 **24 April 2026**: Full code released!
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


### 3️⃣ Training the Routers

Training uses **AdamW**, 25 epochs, **1×10⁻³ LR**, **bf16 precision**, and a **single A100 GPU (40GB)** — taking under 4 hours with only 4K MCTS-derived examples.

#### Supported Models

Modified model files compatible with router training are provided in `train_models/`:

```
train_models/
├── modeling_llama.py
├── modeling_qwen2.py
├── modeling_qwen3.py
└── ...
```

These files insert a `RouterBlock` (Linear-GELU-Linear, hidden dim 128) after each transformer block and expose `init_routers()`, `num_windows`, and `is_static_routing` on the base model class. See `train_models/README.md` for instructions on adapting a new model architecture.

#### Running Router Training

```bash
python train.py \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --run_name drllm-llama-3b \
  --num_epochs 25 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --gradient_accumulation 16
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_id` | — | HuggingFace model path or ID |
| `--run_name` | — | Run name for checkpoints and W&B logging |
| `--num_epochs` | `15` | Number of training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_steps` | `500` | LR warmup steps |
| `--gradient_accumulation` | `16` | Gradient accumulation steps |
| `--num_windows` | `8` | Number of pooling windows for router input |
| `--with_squad` | `False` | Include SQuADv2 data in training |
| `--with_commonsense` | `False` | Include commonsense data in training |

#### Training Details

- **Frozen base**: all base model parameters are frozen; only `model.model.routers` is trained (11M params for 3B models, 0.14% of total weights)
- **Loss**: focal loss with effective-number class rebalancing (`β=0.999`, `γ=2`) to handle the heavy skip/execute/repeat class imbalance
- **Router input**: windowed mean-pooled hidden states from the previous layer (default 8 windows)
- **Teacher forcing**: ground-truth routing labels are used during training to avoid inter-router dependency
- **Optimizer**: AdamW with cosine LR schedule
- **Precision**: bf16
- **Logging**: Weights & Biases (`--report_to wandb`)
- **Checkpoints**: saved to `checkpoints/{run_name}/`

#### Monitored Metrics

During training, the following metrics are logged per step:

| Metric | Description |
|---|---|
| `macro_f1` | Macro F1 across skip/execute/repeat |
| `f1_skip` / `f1_execute` / `f1_repeat` | Per-class F1 scores |
| `acc_skip` / `acc_repeat` | Per-class accuracy for minority classes |
| `avg_layers` | Average number of layers executed per example |
| `routers_loss` | Focal loss on routing decisions |

### 4️⃣ Evaluation with lm-eval-harness

We evaluate Dr.LLM using [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). The same modified model files from `train_models/` are used for evaluation — no additional changes needed.

#### Baseline Evaluation (No Routing)

To evaluate the vanilla model without routing:

```bash
accelerate launch --multi_gpu --num_processes 4 lm_eval \
    --model hf \
    --model_args pretrained="meta-llama/Llama-3.2-3B-Instruct" \
    --tasks arc_challenge,arc_easy,mmlu,aime24,truthfulqa,gsm8k,piqa \
    --batch_size 1
```

#### Dr.LLM Evaluation (With Routing)

To evaluate a trained Dr.LLM checkpoint:

```bash
accelerate launch --num_processes 2 lm_eval \
    --model hf \
    --model_args pretrained=checkpoints/drllm-llama-3b-instruct,dtype=bfloat16,num_windows=8 \
    --tasks arc_challenge,arc_easy,mmlu,aime24,truthfulqa,gsm8k,piqa \
    --batch_size 1 \
    --gen_kwargs max_new_tokens=256 \
    --cache_requests true
```

#### Key Arguments

| Argument | Description |
|---|---|
| `pretrained` | Path to HuggingFace model ID or local Dr.LLM checkpoint |
| `dtype` | Model precision, use `bfloat16` |
| `num_windows` | Number of pooling windows — must match the value used during training |
| `--tasks` | Comma-separated list of benchmarks |
| `--batch_size` | Batch size per device, use `1` for stability |
| `--gen_kwargs` | Generation kwargs, e.g. `max_new_tokens=256` |
| `--cache_requests` | Cache tokenized requests to speed up repeated runs |

#### Supported Benchmarks

| Benchmark | Domain | Split |
|---|---|---|
| `arc_easy` / `arc_challenge` | Logic Reasoning | In-domain |
| `dart` (levels 1–5) | Math Reasoning | In-domain |
| `mmlu` | Factual Knowledge | Out-of-domain |
| `gsm8k` | Grade-school Math | Out-of-domain |
| `aime24` | Competition Math | Out-of-domain |
| `truthfulqa` | Adversarial Factuality | Out-of-domain |
| `gpqa_diamond` | Graduate Reasoning | Out-of-domain |
| `agieval` | Exam Reasoning | Out-of-domain |
| `squadv2` | Reading Comprehension | Out-of-domain |
| `piqa` | Commonsense Reasoning | Out-of-domain |

#### Notes

- All results in the paper use **greedy decoding**, 2048 max tokens, and default `lm-eval-harness` settings.
- `num_windows` must match the value used during training (default: `8`). Mismatches will silently produce incorrect routing decisions.
- Set `is_static_routing=True` in the model to force all decisions to `execute` — useful for sanity-checking that the base model is loaded correctly before evaluating routing.

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

