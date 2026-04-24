# Data Models

This folder contains modified HuggingFace modeling files adapted for Dr.LLM's MCTS-based optimal layer configuration data generation.

## Files

- `modeling_llama.py`
- `modeling_qwen2.py`
- `modeling_qwen3.py`
- *(add your model here)*

## How to Adapt a New Model

To make any decoder-only HuggingFace model compatible with the data generation pipeline, you need to make **two small changes** to the base model class (e.g. `LlamaModel`, `Qwen3Model`):

### 1. Add `layer_indices` in `__init__`

In the model's `__init__`, after the layers are defined, add:

```python
self.layer_indices = list(range(config.num_hidden_layers))
```

### 2. Replace the layer loop in `forward`

Replace the standard loop:

```python
for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    hidden_states = decoder_layer(hidden_states, ...)
```

With an index-driven loop:

```python
decode_layers = [
    self.layers[layer_idx] for layer_idx in self.layer_indices
    if layer_idx < self.config.num_hidden_layers
]
for decoder_layer in decode_layers:
    hidden_states = decoder_layer(hidden_states, ...)
```

That's it. The MCTS search script will manipulate `model.model.layer_indices` externally at runtime to explore different skip/repeat execution paths without modifying model weights.

## How It Works

During data generation, the search script sets `layer_indices` to different path configurations, e.g.:

```python
# Skip layer 3
model.model.layer_indices = [0, 1, 2, 4, 5, ...]

# Repeat layer 7
model.model.layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, ...]
```

Each configuration is evaluated against the task reward (correct/incorrect), and accuracy-preserving or improving paths are retained as supervision signal for the routers.

## Notes

- These files are **only used for offline data generation**, not for training or inference.
- The `ForCausalLM` wrapper and loss functions do not need to be modified.
- Models with sliding window attention (e.g. Qwen2, Qwen3) work as-is since the `attention_type` is fetched from the layer object directly, not the index.
