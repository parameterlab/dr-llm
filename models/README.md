# Models

This folder contains modified HuggingFace modeling files adapted for Dr.LLM router training.

## Files

- `modeling_llama.py`
- `modeling_qwen2.py`
- `modeling_qwen3.py`- *(add your model here)*

## How to Adapt a New Model

To make any decoder-only HuggingFace model compatible with router training, you need to make the following changes.

### 1. Add `RouterBlock` class

Add this class anywhere before the base model class:

```python
class RouterBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, eps=1e-5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
```

### 2. Add routers in `__init__`

In the base model's `__init__` (e.g. `LlamaModel`, `Qwen3Model`), add after the layers are defined:

```python
self.routers = nn.ModuleList(
    [RouterBlock(config.hidden_size) for _ in range(config.num_hidden_layers)]
)
self.num_windows = 8
self.is_static_routing = False
```

### 3. Add `init_routers` method

```python
def init_routers(self):
    for router in self.routers:
        nn.init.xavier_uniform_(router.linear1.weight)
        nn.init.zeros_(router.linear1.bias)
        nn.init.xavier_uniform_(router.linear2.weight)
        nn.init.zeros_(router.linear2.bias)
```

### 4. Modify the layer loop in `forward`

Replace the standard layer loop with the windowed routing loop:

```python
all_decision_logits = []
router_labels = kwargs.get("router_labels", None)
labels = kwargs.get("labels", None)
if labels is not None:
    input_indices = [next((i for i, v in enumerate(x) if v > 0), None) for x in labels.tolist()]
else:
    input_indices = [hidden_states.shape[1] for _ in hidden_states]

for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    input_hiddens = [h[:input_indices[idx], :].unsqueeze(0) for idx, h in enumerate(hidden_states)]
    decision_logits_list = []
    for h in input_hiddens:
        num_windows = min(self.num_windows, h.shape[1])
        window_size = h.shape[1] // num_windows
        windows = h[:, :num_windows * window_size, :].reshape(h.shape[0], num_windows, window_size, -1)
        window_means = windows.mean(dim=2).squeeze(0)
        window_decisions = [self.routers[i](wm.unsqueeze(0)) for wm in window_means]
        window_decisions = torch.stack(window_decisions, dim=0)
        decision_logits = window_decisions.mean(dim=0)
        decision_logits_list.append(decision_logits)
    decision_logits = torch.stack(decision_logits_list, dim=0)
    all_decision_logits.append(decision_logits)

    if self.training and router_labels is not None:
        decisions = router_labels[:, i:i+1]
    else:
        decisions = torch.argmax(decision_logits, dim=-1)
    decision = decisions.squeeze(-1).tolist()[0]
    if self.is_static_routing:
        decision = 1

    for _ in range(decision):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=...,   # your model's mask argument
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
```

### 5. Update the return value

Include `all_decision_logits` in the model output:

```python
return BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=past_key_values if use_cache else None,
    all_decision_logits=all_decision_logits,
)
```

### 6. Add focal loss and metrics in `ForCausalLM.forward`

In the `ForCausalLM` wrapper, extract `routers_labels` from kwargs, compute focal loss against `all_decision_logits`, and return per-class F1, skip/repeat accuracy, and average layers as a `metrics` dict. See any existing model file for the full implementation.

## Training

Routers are trained while the base model is fully frozen. Only `model.model.routers` parameters have `requires_grad=True`. See `train.py` in the root directory for the full training script.

```python
for p in model.model.parameters():
    p.requires_grad = False
for p in model.model.routers.parameters():
    p.requires_grad = True
```

## Notes

- These files are **only used for router training**, not for data generation (see `data_models/`).
- Models with sliding window attention (e.g. Qwen2, Qwen3) work as-is since `attention_type` is fetched from the layer object directly.
- `is_static_routing = True` forces all decisions to `execute` (decision=1), useful for debugging or baseline evaluation.
- `num_windows` defaults to `8` and can be overridden at runtime via `model.model.num_windows = N`.
