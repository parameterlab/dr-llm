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

@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.routers = nn.ModuleList(
            [RouterBlock(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )
        self.num_windows = 8
        self.is_static_routing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def init_routers(self):
        for router in self.routers:
            nn.init.xavier_uniform_(router.linear1.weight)
            nn.init.zeros_(router.linear1.bias)
            nn.init.xavier_uniform_(router.linear2.weight)
            nn.init.zeros_(router.linear2.bias)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
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
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            all_decision_logits=all_decision_logits,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        router_labels = kwargs.get("routers_labels", None)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            router_labels=router_labels,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        text_loss, routers_loss = 0.0, 0.0
        if labels is not None:
            pass

        metrics = None
        router_labels_inference = kwargs.get("router_labels_inference", None)
        if router_labels is not None or router_labels_inference is not None:
            router_labels = router_labels if router_labels is not None else router_labels_inference

            def focal_loss(logits, target, alpha=None, gamma=2.0, eps=1e-8):
                logp = nn.functional.log_softmax(logits, dim=-1)
                p = logp.exp()
                pt = p[torch.arange(target.numel(), device=target.device), target]
                logpt = logp[torch.arange(target.numel(), device=target.device), target]
                if alpha is not None:
                    at = alpha[target]
                    loss = -at * (1 - pt)**gamma * logpt
                else:
                    loss = -(1 - pt)**gamma * logpt
                return loss.mean()

            router_logits = torch.stack(outputs.all_decision_logits, dim=1)
            decision_size = router_logits.shape[-1]
            n_skip, n_execute, n_repeat = 4399, 120956, 1457
            counts = torch.tensor([n_skip, n_execute, n_repeat], dtype=torch.float32)
            beta = 0.999
            eff_num = (1 - beta**counts) / (1 - beta)
            class_w = (1.0 / eff_num)
            class_w = class_w / class_w.sum() * len(class_w)
            class_w = class_w.to(router_logits.device).to(torch.float32)
            alpha = class_w
            routers_loss = focal_loss(router_logits.float().view(-1, 3),
                                    router_labels.view(-1), alpha=alpha, gamma=2.0)

            pred = router_logits.argmax(dim=-1)
            y_true = router_labels.view(-1)
            y_pred = pred.view(-1)
            num_classes = decision_size
            eps = 1e-8

            idx = y_true * num_classes + y_pred
            conf = torch.bincount(idx, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes).to(router_logits.dtype)

            TP = conf.diag()
            FP = conf.sum(dim=0) - TP
            FN = conf.sum(dim=1) - TP

            prec_c = TP / (TP + FP + eps)
            rec_c  = TP / (TP + FN + eps)
            f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
            macro_f1 = f1_c.mean()

            support = conf.sum(dim=1).clamp_min(1.0)
            acc_per_class = TP / support
            acc_skip   = acc_per_class[0]
            acc_repeat = acc_per_class[2]
            skip_count = (router_labels == 0).sum().item()
            repeat_count = (router_labels == 2).sum().item()
            avg_layers = (y_pred.sum() / pred.shape[0])

            metrics = {
                "macro_f1": macro_f1.item(),
                "f1_skip": f1_c[0].item(),
                "f1_execute": f1_c[1].item(),
                "f1_repeat": f1_c[2].item(),
                "acc_skip": acc_skip.item(),
                "acc_repeat": acc_repeat.item(),
                "avg_layers": avg_layers.item(),
                "skip_count": skip_count,
                "repeat_count": repeat_count,
                "routers_loss": routers_loss.item(),
                "text_loss": text_loss.item() if isinstance(text_loss, torch.Tensor) else text_loss,
            }

        loss = routers_loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            metrics=metrics,
        )
