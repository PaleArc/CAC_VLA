# ruff: noqa: F821, RET504, SIM102

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import siglip as _siglip
import openpi.models.gemma_oat as _gemma_oat
from openpi.models.pi0_oat_config import OATObservation
from openpi.models.pi0_oat_config import Pi0OatConfig
from openpi.models.pi0_oat_config import preprocess_oat_observation
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class ExpertMemoryQueryResampler(nnx.Module):
    def __init__(self, width: int, num_queries: int, rngs: nnx.Rngs):
        self.width = width
        self.num_queries = num_queries
        self.query_embeddings = nnx.Param(
            jax.random.normal(rngs.params(), (num_queries, width), dtype=jnp.float32) * (width**-0.5)
        )
        self.query_norm = nnx.LayerNorm(width, rngs=rngs)
        self.memory_norm = nnx.LayerNorm(width, rngs=rngs)
        self.q_proj = nnx.Linear(width, width, rngs=rngs)
        self.k_proj = nnx.Linear(width, width, rngs=rngs)
        self.v_proj = nnx.Linear(width, width, rngs=rngs)
        self.out_proj = nnx.Linear(width, width, rngs=rngs)

    def __call__(
        self,
        memory: at.Float[at.Array, "b s d"],
        memory_mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        batch_size = memory.shape[0]
        query_tokens = jnp.broadcast_to(
            self.query_embeddings.value[None, :, :], (batch_size, self.num_queries, self.width)
        )
        query_tokens = self.query_norm(query_tokens.astype(jnp.float32))
        memory = self.memory_norm(jnp.asarray(memory, dtype=jnp.float32))

        q = self.q_proj(query_tokens)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        logits = jnp.einsum("bqd,bsd->bqs", q, k, preferred_element_type=jnp.float32) * (self.width**-0.5)
        if memory_mask is None:
            memory_mask = jnp.ones(memory.shape[:2], dtype=bool)
        else:
            memory_mask = jnp.asarray(memory_mask, dtype=bool)
        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(memory_mask[:, None, :], logits, big_neg)
        probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        attended = jnp.einsum("bqs,bsd->bqd", probs, v, preferred_element_type=jnp.float32)
        updated_queries = query_tokens + self.out_proj(attended)
        has_memory = jnp.any(memory_mask, axis=1, keepdims=True)[..., None]
        return jnp.where(has_memory, updated_queries, query_tokens)


class Pi0Oat(_model.BaseModel):
    def __init__(self, config: Pi0OatConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        if not config.pi05:
            raise ValueError("Pi0Oat currently only supports pi05=True.")

        self.pi05 = config.pi05
        self.use_oat_latent_alignment = bool(config.use_oat_latent_alignment)
        self.lambda_latent = float(config.lambda_latent)
        self.oat_num_queries = int(config.oat_num_queries or 0)
        self.oat_latent_dim = int(config.oat_latent_dim or 0)
        self.use_oat_alignment_target_stop_gradient = bool(config.use_oat_alignment_target_stop_gradient)
        self.use_oat_latent_reconstruction = bool(config.use_oat_latent_reconstruction)
        self.lambda_oat_recon = float(config.lambda_oat_recon)
        self.use_oat_pooled_cond = bool(config.use_oat_pooled_cond)
        self.oat_pooled_cond_source_train = str(config.oat_pooled_cond_source_train)
        self.oat_pooled_cond_source_infer = str(config.oat_pooled_cond_source_infer)
        self.oat_pooled_cond_dropout_rate = float(config.oat_pooled_cond_dropout_rate)
        self.oat_queries_visible_to_action_expert = bool(config.oat_queries_visible_to_action_expert)
        self.oat_latent_cross_attention_to_expert = bool(config.oat_latent_cross_attention_to_expert)
        self.oat_expert_memory_source_train = str(config.oat_expert_memory_source_train)
        self.oat_expert_memory_source_infer = str(config.oat_expert_memory_source_infer)
        self.oat_expert_memory_projected_oat_train_prob = float(config.oat_expert_memory_projected_oat_train_prob)
        self.oat_expert_memory_dropout_rate = float(config.oat_expert_memory_dropout_rate)
        self.oat_expert_memory_time_schedule = str(config.oat_expert_memory_time_schedule)
        self.oat_expert_memory_schedule_domain = str(config.oat_expert_memory_schedule_domain)
        self.oat_expert_memory_min_tokens = int(config.oat_expert_memory_min_tokens)
        self.oat_expert_memory_token_shift = int(config.oat_expert_memory_token_shift)
        self.use_oat_expert_memory_query_resampler = bool(config.use_oat_expert_memory_query_resampler)
        self.oat_expert_memory_num_queries = int(config.oat_expert_memory_num_queries)
        self.use_internal_query_memory_for_expert_attn = bool(config.use_internal_query_memory_for_expert_attn)
        self.use_oat_cross_attn_residual_gate = bool(config.use_oat_cross_attn_residual_gate)
        self.use_oat_cross_attn_hidden_gate = bool(config.use_oat_cross_attn_hidden_gate)
        self.oat_cross_attn_hidden_gate_bias_init = float(config.oat_cross_attn_hidden_gate_bias_init)
        self.use_oat_cross_attn_context_gate = bool(config.use_oat_cross_attn_context_gate)
        self.oat_cross_attn_context_gate_bias_init = float(config.oat_cross_attn_context_gate_bias_init)
        self.oat_cross_attn_context_gate_inputs = tuple(config.oat_cross_attn_context_gate_inputs)
        self.use_oat_cross_attn_kv_soft_gate = bool(config.use_oat_cross_attn_kv_soft_gate)
        self.oat_cross_attn_kv_soft_gate_bias_init = float(config.oat_cross_attn_kv_soft_gate_bias_init)
        self.oat_cross_attn_last_n_layers = config.oat_cross_attn_last_n_layers
        self.oat_alignment_target_space = str(config.oat_alignment_target_space)
        self.oat_expert_memory_input_space = str(config.oat_expert_memory_input_space)
        self.oat_query_hidden_noise_std = float(config.oat_query_hidden_noise_std)
        self.oat_expert_memory_noise_std = float(config.oat_expert_memory_noise_std)
        self.oat_expert_memory_noise_train_prob = float(config.oat_expert_memory_noise_train_prob)

        paligemma_config = _gemma_oat.get_config(config.paligemma_variant)
        action_expert_config = _gemma_oat.get_config(config.action_expert_variant)
        self.paligemma_depth = paligemma_config.depth
        self.paligemma_width = paligemma_config.width
        if self.oat_cross_attn_last_n_layers is not None:
            if self.oat_cross_attn_last_n_layers <= 0 or self.oat_cross_attn_last_n_layers > self.paligemma_depth:
                raise ValueError(
                    "oat_cross_attn_last_n_layers must be in "
                    f"[1, {self.paligemma_depth}], got {self.oat_cross_attn_last_n_layers}."
                )
        llm = nnx_bridge.ToNNX(
            _gemma_oat.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                latent_cross_attention_to_expert=self.oat_latent_cross_attention_to_expert,
                use_cross_attn_residual_gate=self.use_oat_cross_attn_residual_gate,
                use_cross_attn_hidden_gate=self.use_oat_cross_attn_hidden_gate,
                cross_attn_hidden_gate_bias_init=self.oat_cross_attn_hidden_gate_bias_init,
                use_cross_attn_context_gate=self.use_oat_cross_attn_context_gate,
                cross_attn_context_gate_bias_init=self.oat_cross_attn_context_gate_bias_init,
                cross_attn_context_gate_inputs=self.oat_cross_attn_context_gate_inputs,
                use_cross_attn_kv_soft_gate=self.use_oat_cross_attn_kv_soft_gate,
                cross_attn_kv_soft_gate_bias_init=self.oat_cross_attn_kv_soft_gate_bias_init,
                cross_attn_last_n_layers=self.oat_cross_attn_last_n_layers,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if self.use_oat_latent_alignment:
            self.latent_query_embeddings = nnx.Param(
                jax.random.normal(rngs.params(), (self.oat_num_queries, paligemma_config.width), dtype=jnp.float32)
                * (paligemma_config.width**-0.5)
            )
            self.oat_to_hidden_proj = nnx.Linear(self.oat_latent_dim, paligemma_config.width, rngs=rngs)
            if self.use_oat_latent_reconstruction:
                self.oat_recon_proj = nnx.Linear(paligemma_config.width, self.oat_latent_dim, rngs=rngs)
            self.query_memory_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.query_memory_proj = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            if self._uses_raw_oat_prediction_head():
                self.query_oat_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
                self.query_oat_proj = nnx.Linear(paligemma_config.width, self.oat_latent_dim, rngs=rngs)
            if self.use_oat_expert_memory_query_resampler:
                self.expert_memory_resampler = ExpertMemoryQueryResampler(
                    paligemma_config.width,
                    self.oat_expert_memory_num_queries,
                    rngs,
                )
            if self.use_oat_pooled_cond:
                expert_width = self.action_in_proj.out_features
                self.oat_cond_mlp_in = nnx.Linear(paligemma_config.width, expert_width, rngs=rngs)
                self.oat_cond_mlp_out = nnx.Linear(expert_width, expert_width, rngs=rngs)

        self.deterministic = True

    def _get_latent_query_tokens(
        self, batch_size: int
    ) -> tuple[at.Float[at.Array, "b q d"], at.Bool[at.Array, "b q"], at.Bool[at.Array, "b q"]]:
        hidden_dim = self.latent_query_embeddings.value.shape[-1]
        query_tokens = jnp.broadcast_to(
            self.latent_query_embeddings.value[None, :, :],
            (batch_size, self.oat_num_queries, hidden_dim),
        )
        query_mask = jnp.ones((batch_size, self.oat_num_queries), dtype=bool)
        query_ar_mask = jnp.ones((batch_size, self.oat_num_queries), dtype=bool)
        return query_tokens, query_mask, query_ar_mask

    def _uses_query_hidden_expert_memory(self, *, train: bool) -> bool:
        if not self.oat_latent_cross_attention_to_expert:
            return False
        source = self.oat_expert_memory_source_train if train else self.oat_expert_memory_source_infer
        return source in {"query_hidden", "mixed"}

    def _uses_mixed_expert_memory(self, *, train: bool) -> bool:
        return train and self.oat_latent_cross_attention_to_expert and self.oat_expert_memory_source_train == "mixed"

    def _uses_internal_query_hidden_expert_memory(self, *, train: bool) -> bool:
        return self.use_internal_query_memory_for_expert_attn and self._uses_query_hidden_expert_memory(train=train)

    def _uses_projected_oat_expert_memory(self, *, train: bool) -> bool:
        if not self.oat_latent_cross_attention_to_expert:
            return False
        return train and self.oat_expert_memory_source_train in {"projected_oat", "mixed"}

    def _uses_raw_oat_prediction_head(self) -> bool:
        return self.oat_alignment_target_space == "raw" or (
            self.oat_expert_memory_input_space == "raw"
            and (
                self._uses_query_hidden_expert_memory(train=True) or self._uses_query_hidden_expert_memory(train=False)
            )
        )

    def _get_latent_query_indices(
        self, prefix_mask: at.Bool[at.Array, "b s"], *, train: bool
    ) -> at.Int[at.Array, "b q"] | None:
        if not (self._uses_query_hidden_expert_memory(train=train) or self._uses_query_hidden_pooled_cond(train=train)):
            return None
        query_start = prefix_mask.shape[1] - self.oat_num_queries
        query_positions = jnp.arange(query_start, prefix_mask.shape[1], dtype=jnp.int32)
        return jnp.broadcast_to(query_positions[None, :], (prefix_mask.shape[0], self.oat_num_queries))

    def _project_oat_to_hidden(self, observation: OATObservation) -> at.Float[at.Array, "b q d"]:
        if observation.oat_latents is None:
            raise ValueError("OAT alignment requires `oat_latents`.")
        return self._project_raw_oat_to_hidden(jnp.asarray(observation.oat_latents, dtype=jnp.float32))

    def _project_raw_oat_to_hidden(
        self,
        oat_latents: at.Float[at.Array, "b q d"],
    ) -> at.Float[at.Array, "b q d"]:
        return self.oat_to_hidden_proj(jnp.asarray(oat_latents, dtype=jnp.float32))

    def _add_scale_aware_gaussian_noise(
        self,
        x: at.Float[at.Array, "b q d"],
        rng: at.KeyArrayLike | None,
        std: float,
        *,
        train: bool,
    ) -> at.Float[at.Array, "b q d"]:
        if not train or rng is None or std <= 0.0:
            return x
        x_f32 = jnp.asarray(x, dtype=jnp.float32)
        feature_scale = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + 1e-6))
        noise = jax.random.normal(rng, x_f32.shape, dtype=jnp.float32)
        return (x_f32 + std * feature_scale * noise).astype(x.dtype)

    def _add_scale_aware_gaussian_noise_with_train_prob(
        self,
        x: at.Float[at.Array, "b q d"],
        rng: at.KeyArrayLike | None,
        std: float,
        prob: float,
        *,
        train: bool,
    ) -> at.Float[at.Array, "b q d"]:
        if not train or rng is None or std <= 0.0 or prob <= 0.0:
            return x
        if prob >= 1.0:
            return self._add_scale_aware_gaussian_noise(x, rng, std, train=train)

        noise_rng, gate_rng = jax.random.split(rng)
        noisy = self._add_scale_aware_gaussian_noise(x, noise_rng, std, train=train)
        apply_noise = jax.random.bernoulli(gate_rng, prob, (x.shape[0],))
        return jnp.where(apply_noise[:, None, None], noisy, x)

    def _project_query_hidden_to_memory(
        self,
        query_hidden: at.Float[at.Array, "b q d"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        query_hidden = jnp.asarray(query_hidden, dtype=jnp.float32)
        query_hidden = self._add_scale_aware_gaussian_noise(
            query_hidden,
            rng,
            self.oat_query_hidden_noise_std,
            train=train,
        )
        query_hidden = self.query_memory_norm(query_hidden)
        return self.query_memory_proj(query_hidden)

    def _project_query_hidden_to_raw_oat(
        self,
        query_hidden: at.Float[at.Array, "b q d"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        query_hidden = jnp.asarray(query_hidden, dtype=jnp.float32)
        query_hidden = self._add_scale_aware_gaussian_noise(
            query_hidden,
            rng,
            self.oat_query_hidden_noise_std,
            train=train,
        )
        query_hidden = self.query_oat_norm(query_hidden)
        return self.query_oat_proj(query_hidden)

    def _get_expert_memory_token_count(self) -> int:
        if self.use_oat_expert_memory_query_resampler:
            return self.oat_expert_memory_num_queries
        return self.oat_num_queries

    def _resample_expert_memory(
        self,
        memory: at.Float[at.Array, "b s d"] | at.Float[at.Array, "l b s d"],
        memory_mask: at.Bool[at.Array, "b s"] | at.Bool[at.Array, "l b s"] | None = None,
    ) -> at.Float[at.Array, "b q d"] | at.Float[at.Array, "l b q d"]:
        if not self.use_oat_expert_memory_query_resampler:
            return memory
        if memory.ndim == 3:
            return self.expert_memory_resampler(memory, memory_mask)
        if memory.ndim != 4:
            raise ValueError(f"Unsupported expert memory rank: {memory.ndim}")

        layers, batch_size, _, width = memory.shape
        flat_memory = memory.reshape(layers * batch_size, memory.shape[2], width)
        flat_mask = (
            None
            if memory_mask is None
            else jnp.asarray(memory_mask, dtype=bool).reshape(layers * batch_size, memory.shape[2])
        )
        resampled = self.expert_memory_resampler(flat_memory, flat_mask)
        return resampled.reshape(layers, batch_size, self.oat_expert_memory_num_queries, width)

    def _get_internal_query_memory_adapter_kwargs(self, *, train: bool) -> dict[str, jax.Array | float]:
        if not self._uses_internal_query_hidden_expert_memory(train=train):
            return {}
        return {
            "query_memory_norm_scale": jnp.asarray(self.query_memory_norm.scale.value, dtype=jnp.float32),
            "query_memory_norm_bias": jnp.asarray(self.query_memory_norm.bias.value, dtype=jnp.float32),
            "query_memory_norm_epsilon": float(self.query_memory_norm.epsilon),
            "query_memory_proj_kernel": jnp.asarray(self.query_memory_proj.kernel.value, dtype=jnp.float32),
            "query_memory_proj_bias": jnp.asarray(self.query_memory_proj.bias.value, dtype=jnp.float32),
        }

    def _uses_query_hidden_pooled_cond(self, *, train: bool) -> bool:
        if not self.use_oat_pooled_cond:
            return False
        source = self.oat_pooled_cond_source_train if train else self.oat_pooled_cond_source_infer
        return source == "query_hidden"

    def _uses_projected_oat_pooled_cond(self, *, train: bool) -> bool:
        if not self.use_oat_pooled_cond:
            return False
        source = self.oat_pooled_cond_source_train if train else self.oat_pooled_cond_source_infer
        return source == "projected_oat"

    def _get_pooled_cond_query_mask(self, batch_size: int) -> at.Bool[at.Array, "b q"]:
        return jnp.ones((batch_size, self.oat_num_queries), dtype=bool)

    def _encode_oat_cond_tokens(
        self,
        cond_tokens: at.Float[at.Array, "b q d"],
        cond_mask: at.Bool[at.Array, "b q"],
    ) -> at.Float[at.Array, "b emb"]:
        mask = jnp.asarray(cond_mask, dtype=bool)[..., None]
        cond_tokens = jnp.asarray(cond_tokens, dtype=jnp.float32)
        pooled = jnp.sum(cond_tokens * mask.astype(cond_tokens.dtype), axis=1)
        denom = jnp.clip(jnp.sum(mask.astype(cond_tokens.dtype), axis=1), a_min=1.0)
        pooled = pooled / denom
        has_cond = jnp.any(cond_mask, axis=1, keepdims=True)

        oat_cond = self.oat_cond_mlp_in(pooled)
        oat_cond = nnx.swish(oat_cond)
        oat_cond = self.oat_cond_mlp_out(oat_cond)
        oat_cond = nnx.swish(oat_cond)
        return jnp.where(has_cond, oat_cond, jnp.zeros_like(oat_cond))

    def _sample_pooled_cond_gate(
        self,
        rng: at.KeyArrayLike,
        batch_size: int,
        *,
        train: bool,
    ) -> at.Bool[at.Array, "b"]:
        if not self.use_oat_pooled_cond or not train or self.oat_pooled_cond_dropout_rate <= 0.0:
            return jnp.ones((batch_size,), dtype=bool)
        keep_prob = 1.0 - self.oat_pooled_cond_dropout_rate
        return jax.random.bernoulli(rng, keep_prob, (batch_size,))

    def _build_projected_oat_expert_cross_attn_memory(
        self, oat_hidden_targets: at.Float[at.Array, "b q d"], *, train: bool
    ) -> at.Float[at.Array, "l b q d"] | None:
        if not self._uses_projected_oat_expert_memory(train=train):
            return None
        return self._build_expert_cross_attn_memory(oat_hidden_targets)

    def _build_query_hidden_expert_cross_attn_memory(
        self, query_memory: at.Float[at.Array, "b q d"], *, train: bool
    ) -> at.Float[at.Array, "l b q d"] | None:
        if not self._uses_query_hidden_expert_memory(train=train):
            return None
        if self.oat_expert_memory_input_space == "raw":
            return self._build_raw_oat_expert_cross_attn_memory(query_memory, train=train)
        return self._build_expert_cross_attn_memory(query_memory)

    def _build_raw_oat_expert_cross_attn_memory(
        self, raw_oat_memory: at.Float[at.Array, "b q d"], *, train: bool
    ) -> at.Float[at.Array, "l b q d"] | None:
        del train
        return self._build_expert_cross_attn_memory(self._project_raw_oat_to_hidden(raw_oat_memory))

    def _build_expert_cross_attn_memory(self, memory: at.Float[at.Array, "b q d"]) -> at.Float[at.Array, "l b q d"]:
        memory = self._resample_expert_memory(memory)
        return jnp.broadcast_to(memory[None, ...], (self.paligemma_depth, *memory.shape))

    def _mix_expert_cross_attn_memory(
        self,
        projected_oat_memory: at.Float[at.Array, "l b q d"],
        query_hidden_memory: at.Float[at.Array, "l b q d"],
        rng: at.KeyArrayLike,
        batch_size: int,
        *,
        train: bool,
    ) -> at.Float[at.Array, "l b q d"]:
        if not self._uses_mixed_expert_memory(train=train):
            return projected_oat_memory
        use_projected_oat = jax.random.bernoulli(
            rng,
            self.oat_expert_memory_projected_oat_train_prob,
            (batch_size,),
        )
        return jnp.where(use_projected_oat[None, :, None, None], projected_oat_memory, query_hidden_memory)

    def _sample_cross_attn_gate(
        self,
        rng: at.KeyArrayLike,
        batch_size: int,
        *,
        train: bool,
    ) -> at.Bool[at.Array, "b"]:
        if not self.oat_latent_cross_attention_to_expert or not train or self.oat_expert_memory_dropout_rate <= 0.0:
            return jnp.ones((batch_size,), dtype=bool)
        keep_prob = 1.0 - self.oat_expert_memory_dropout_rate
        return jax.random.bernoulli(rng, keep_prob, (batch_size,))

    def _uses_expert_memory_time_schedule(self, *, train: bool) -> bool:
        del train
        return self.oat_latent_cross_attention_to_expert and self.oat_expert_memory_time_schedule != "none"

    def _compute_expert_memory_schedule_progress(
        self,
        timestep: at.Float[at.Array, "b"],
    ) -> at.Float[at.Array, "b"]:
        timestep = jnp.asarray(timestep, dtype=jnp.float32)
        if self.oat_expert_memory_schedule_domain == "t":
            return jnp.clip(1.0 - timestep, 0.0, 1.0)
        if self.oat_expert_memory_schedule_domain != "log_snr":
            raise ValueError(
                f"Unsupported oat_expert_memory_schedule_domain={self.oat_expert_memory_schedule_domain!r}"
            )

        # This repository uses flow matching: x_t = t * noise + (1 - t) * actions.
        # We therefore interpret alpha=1-t and sigma=t, then convert logSNR into a
        # monotonic [0, 1] progress scalar via sigmoid(logSNR)=SNR/(1+SNR).
        alpha = jnp.clip(1.0 - timestep, 1e-6, 1.0)
        sigma = jnp.clip(timestep, 1e-6, 1.0)
        log_snr = jnp.log(jnp.square(alpha)) - jnp.log(jnp.square(sigma))
        return jax.nn.sigmoid(log_snr)

    def _build_expert_cross_attn_memory_mask(
        self,
        timestep: at.Float[at.Array, "b"],
        *,
        train: bool,
    ) -> at.Bool[at.Array, "b q"] | None:
        if not self._uses_expert_memory_time_schedule(train=train):
            return None
        if self.oat_expert_memory_time_schedule not in {"ceil_linear", "ceil_sin", "staged_4_6_8"}:
            raise ValueError(f"Unsupported oat_expert_memory_time_schedule={self.oat_expert_memory_time_schedule!r}")
        progress = self._compute_expert_memory_schedule_progress(timestep)
        memory_tokens = self._get_expert_memory_token_count()

        if self.oat_expert_memory_time_schedule == "staged_4_6_8":
            raw_k = jnp.where(
                progress < (1.0 / 3.0),
                4,
                jnp.where(progress < (2.0 / 3.0), 6, 8),
            ).astype(jnp.int32)
        elif self.oat_expert_memory_time_schedule == "ceil_linear":
            raw_k = jnp.ceil(progress * memory_tokens).astype(jnp.int32)
        else:
            raw_k = jnp.ceil(memory_tokens * jnp.sin(0.5 * jnp.pi * progress)).astype(jnp.int32)

        if self.oat_expert_memory_time_schedule == "staged_4_6_8":
            k = jnp.clip(raw_k, 1, memory_tokens)
        else:
            old_k = jnp.clip(raw_k, self.oat_expert_memory_min_tokens, memory_tokens)
            k = jnp.clip(old_k + self.oat_expert_memory_token_shift, 1, memory_tokens)

        token_indices = jnp.arange(memory_tokens, dtype=jnp.int32)[None, :]
        return token_indices < k[:, None]

    def _build_prefix_attention_mask(
        self,
        base_mask: at.Bool[at.Array, "b s"],
        query_mask: at.Bool[at.Array, "b q"] | None,
    ) -> at.Bool[at.Array, "b _t _s"]:
        if query_mask is None:
            return _pi0.make_attn_mask(base_mask, jnp.zeros_like(base_mask, dtype=bool))

        batch_size = base_mask.shape[0]
        base_len = base_mask.shape[1]
        prefix_len = base_len + query_mask.shape[1]
        attn_mask = jnp.zeros((batch_size, prefix_len, prefix_len), dtype=bool)

        base_attn = _pi0.make_attn_mask(base_mask, jnp.zeros_like(base_mask, dtype=bool))
        attn_mask = attn_mask.at[:, :base_len, :base_len].set(base_attn)

        query_to_base = base_mask[:, None, :]
        query_to_query = _pi0.make_attn_mask(query_mask, query_mask)
        attn_mask = attn_mask.at[:, base_len:, :base_len].set(query_to_base)
        attn_mask = attn_mask.at[:, base_len:, base_len:].set(query_to_query)
        return attn_mask

    @at.typecheck
    def embed_prefix(
        self, obs: OATObservation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s s"]]:
        tokens = []
        image_mask = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            image_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        text_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        text_mask = obs.tokenized_prompt_mask

        tokens.append(text_tokens)
        base_tokens = jnp.concatenate(tokens, axis=1)
        base_mask = jnp.concatenate([*image_mask, text_mask], axis=1)

        query_mask = None
        if self.use_oat_latent_alignment:
            query_tokens, query_mask, _ = self._get_latent_query_tokens(obs.tokenized_prompt.shape[0])
            base_tokens = jnp.concatenate([base_tokens, query_tokens], axis=1)
            base_mask = jnp.concatenate([base_mask, query_mask], axis=1)

        prefix_attn_mask = self._build_prefix_attention_mask(
            jnp.concatenate([*image_mask, text_mask], axis=1),
            query_mask,
        )
        return base_tokens, base_mask, prefix_attn_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: OATObservation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        oat_cond: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "s"],
        at.Float[at.Array, "b emb"],
    ]:
        del obs
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb if oat_cond is None else time_emb + oat_cond
        input_mask = jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + ([False] * (self.action_horizon - 1)))
        return action_expert_tokens, input_mask, ar_mask, adarms_cond

    def _build_prefix_action_mask(self, prefix_mask: at.Bool[at.Array, "b s"]) -> at.Bool[at.Array, "b s"]:
        if not self.use_oat_latent_alignment or self.oat_queries_visible_to_action_expert:
            return prefix_mask
        hidden_query_mask = jnp.concatenate(
            [
                jnp.zeros((prefix_mask.shape[0], prefix_mask.shape[1] - self.oat_num_queries), dtype=bool),
                jnp.ones((prefix_mask.shape[0], self.oat_num_queries), dtype=bool),
            ],
            axis=1,
        )
        return jnp.logical_and(prefix_mask, jnp.logical_not(hidden_query_mask))

    def _uses_cross_attn_context_gate(self) -> bool:
        return bool(
            self.use_oat_cross_attn_residual_gate
            and self.use_oat_cross_attn_hidden_gate
            and self.use_oat_cross_attn_context_gate
        )

    def _uses_cross_attn_prefix_context(self) -> bool:
        return self._uses_cross_attn_context_gate() or self.use_oat_cross_attn_kv_soft_gate

    def _build_prefix_context_mask(self, prefix_mask: at.Bool[at.Array, "b s"]) -> at.Bool[at.Array, "b s"]:
        if not self.use_oat_latent_alignment:
            return prefix_mask
        hidden_query_mask = jnp.concatenate(
            [
                jnp.zeros((prefix_mask.shape[0], prefix_mask.shape[1] - self.oat_num_queries), dtype=bool),
                jnp.ones((prefix_mask.shape[0], self.oat_num_queries), dtype=bool),
            ],
            axis=1,
        )
        return jnp.logical_and(prefix_mask, jnp.logical_not(hidden_query_mask))

    def _build_prefix_context_summary(
        self,
        prefix_hidden: at.Float[at.Array, "b s d"],
        prefix_context_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Float[at.Array, "b d"] | None:
        if prefix_context_mask is None:
            return None
        weights = prefix_context_mask[..., None].astype(prefix_hidden.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        return jnp.sum(prefix_hidden * weights, axis=1) / denom

    def _build_combined_attention_mask(
        self,
        prefix_attn: at.Bool[at.Array, "b p p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"],
        suffix_ar_mask: at.Bool[at.Array, "s"],
    ) -> at.Bool[at.Array, "b _t _s"]:
        batch_size, prefix_len = prefix_mask_action.shape
        suffix_len = suffix_mask.shape[1]
        combined = jnp.zeros((batch_size, prefix_len + suffix_len, prefix_len + suffix_len), dtype=bool)
        combined = combined.at[:, :prefix_len, :prefix_len].set(prefix_attn)

        prefix_ar_mask_action = jnp.zeros_like(prefix_mask_action, dtype=bool)
        input_mask = jnp.concatenate([prefix_mask_action, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask_action, jnp.broadcast_to(suffix_ar_mask, suffix_mask.shape)], axis=1)
        action_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        combined = combined.at[:, prefix_len:, :].set(action_mask[:, prefix_len:, :])
        return combined

    def _build_combined_positions(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"],
    ) -> at.Int[at.Array, "b _t"]:
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1, keepdims=True) + jnp.cumsum(suffix_mask, axis=-1) - 1
        return jnp.concatenate([prefix_positions, suffix_positions], axis=1).astype(jnp.int32)

    def _compute_latent_alignment_loss(
        self,
        oat_targets: at.Float[at.Array, "b q d"],
        latent_mask: at.Bool[at.Array, "b q"],
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        *,
        train: bool = False,
        query_hidden_noise_rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if self.oat_alignment_target_space == "raw":
            query_prediction = self._extract_query_raw_oat_from_prefix_logits(
                prefix_pre_logits,
                train=train,
                rng=query_hidden_noise_rng,
            )
        else:
            query_prediction = self._extract_query_memory_from_prefix_logits(
                prefix_pre_logits,
                train=train,
                rng=query_hidden_noise_rng,
            )
        target_hidden = jnp.asarray(oat_targets, dtype=query_prediction.dtype)
        if self.use_oat_alignment_target_stop_gradient:
            target_hidden = jax.lax.stop_gradient(target_hidden)
        latent_mask = jnp.asarray(latent_mask, dtype=bool)

        diff = query_prediction - target_hidden
        abs_diff = jnp.abs(diff)
        per_dim = jnp.where(abs_diff < 1.0, 0.5 * jnp.square(diff), abs_diff - 0.5)
        per_slot = jnp.mean(per_dim, axis=-1)
        denom = jnp.maximum(jnp.sum(latent_mask, axis=-1), 1.0)
        per_sample = jnp.sum(per_slot * latent_mask.astype(per_slot.dtype), axis=-1) / denom
        metrics = {
            "latent_align_loss": jnp.sum(per_slot * latent_mask.astype(per_slot.dtype))
            / jnp.maximum(jnp.sum(latent_mask), 1.0)
        }
        return per_sample, metrics

    def _compute_oat_reconstruction_loss(
        self,
        oat_hidden_targets: at.Float[at.Array, "b q d"],
        oat_latents: at.Float[at.Array, "b q d"],
        latent_mask: at.Bool[at.Array, "b q"],
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        recon = self.oat_recon_proj(jnp.asarray(oat_hidden_targets, dtype=jnp.float32))
        target = jax.lax.stop_gradient(jnp.asarray(oat_latents, dtype=recon.dtype))
        latent_mask = jnp.asarray(latent_mask, dtype=bool)

        per_slot = jnp.mean(jnp.square(recon - target), axis=-1)
        denom = jnp.maximum(jnp.sum(latent_mask, axis=-1), 1.0)
        per_sample = jnp.sum(per_slot * latent_mask.astype(per_slot.dtype), axis=-1) / denom
        metrics = {
            "oat_recon_loss": jnp.sum(per_slot * latent_mask.astype(per_slot.dtype))
            / jnp.maximum(jnp.sum(latent_mask), 1.0)
        }
        return per_sample, metrics

    def _combine_oat_losses(
        self,
        action_loss: at.Float[at.Array, "*b"],
        metrics: dict[str, at.Array],
        *,
        latent_loss: at.Float[at.Array, "*b"] | None = None,
        latent_metrics: dict[str, at.Array] | None = None,
        oat_recon_loss: at.Float[at.Array, "*b"] | None = None,
        oat_recon_metrics: dict[str, at.Array] | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        total_loss = action_loss
        if latent_loss is not None:
            total_loss = total_loss + self.lambda_latent * latent_loss
        if oat_recon_loss is not None:
            total_loss = total_loss + self.lambda_oat_recon * oat_recon_loss
        if latent_metrics is not None:
            metrics.update(latent_metrics)
        if oat_recon_metrics is not None:
            metrics.update(oat_recon_metrics)
        return total_loss, metrics

    def _extract_query_hidden_from_prefix_logits(
        self,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
    ) -> at.Float[at.Array, "b q d"]:
        return prefix_pre_logits[:, -self.oat_num_queries :]

    def _extract_query_memory_from_prefix_logits(
        self,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        query_hidden = self._extract_query_hidden_from_prefix_logits(prefix_pre_logits)
        return self._project_query_hidden_to_memory(query_hidden, train=train, rng=rng)

    def _extract_query_raw_oat_from_prefix_logits(
        self,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        query_hidden = self._extract_query_hidden_from_prefix_logits(prefix_pre_logits)
        return self._project_query_hidden_to_raw_oat(query_hidden, train=train, rng=rng)

    def _extract_query_expert_memory_from_prefix_logits(
        self,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        if self.oat_expert_memory_input_space == "raw":
            return self._extract_query_raw_oat_from_prefix_logits(prefix_pre_logits, train=train, rng=rng)
        return self._extract_query_memory_from_prefix_logits(prefix_pre_logits, train=train, rng=rng)

    def _compute_loss_components(
        self,
        rng: at.KeyArrayLike,
        observation: OATObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng, dropout_rng, pooled_cond_dropout_rng = jax.random.split(rng, 5)
        query_hidden_noise_rng = jax.random.fold_in(rng, 5)
        expert_memory_noise_rng = jax.random.fold_in(rng, 6)
        expert_memory_mix_rng = jax.random.fold_in(rng, 7)
        observation = preprocess_oat_observation(preprocess_rng, observation, train=train)

        if self.use_oat_latent_alignment and (observation.oat_latents is None or observation.oat_latent_mask is None):
            raise ValueError("use_oat_latent_alignment=True requires batch fields `oat_latents` and `oat_latent_mask`.")

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_attn_mask = self.embed_prefix(observation)
        use_prefix_context = self._uses_cross_attn_prefix_context()
        prefix_context_mask = self._build_prefix_context_mask(prefix_mask) if use_prefix_context else None
        oat_latents_clean = (
            jnp.asarray(observation.oat_latents, dtype=jnp.float32) if self.use_oat_latent_alignment else None
        )
        oat_hidden_targets_clean = (
            self._project_raw_oat_to_hidden(oat_latents_clean) if oat_latents_clean is not None else None
        )
        if oat_latents_clean is not None and self.oat_expert_memory_input_space == "raw":
            oat_latents_for_expert = self._add_scale_aware_gaussian_noise_with_train_prob(
                oat_latents_clean,
                expert_memory_noise_rng,
                self.oat_expert_memory_noise_std,
                self.oat_expert_memory_noise_train_prob,
                train=train,
            )
            oat_hidden_targets_for_expert = self._project_raw_oat_to_hidden(oat_latents_for_expert)
        elif oat_hidden_targets_clean is not None:
            oat_hidden_targets_for_expert = self._add_scale_aware_gaussian_noise_with_train_prob(
                oat_hidden_targets_clean,
                expert_memory_noise_rng,
                self.oat_expert_memory_noise_std,
                self.oat_expert_memory_noise_train_prob,
                train=train,
            )
        else:
            oat_hidden_targets_for_expert = None
        alignment_targets = oat_latents_clean if self.oat_alignment_target_space == "raw" else oat_hidden_targets_clean
        oat_recon_loss = None
        oat_recon_metrics = None
        if self.use_oat_latent_reconstruction:
            oat_recon_loss, oat_recon_metrics = self._compute_oat_reconstruction_loss(
                oat_hidden_targets_clean,
                observation.oat_latents,
                observation.oat_latent_mask,
            )

        pooled_cond_gate = self._sample_pooled_cond_gate(
            pooled_cond_dropout_rng,
            actions.shape[0],
            train=train,
        )

        if self.use_oat_pooled_cond and self._uses_projected_oat_pooled_cond(train=True):
            oat_cond = self._encode_oat_cond_tokens(
                oat_hidden_targets_clean,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
            )
            oat_cond = oat_cond * pooled_cond_gate[:, None].astype(oat_cond.dtype)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, oat_cond=oat_cond
            )
            prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
            combined_mask = self._build_combined_attention_mask(
                prefix_attn_mask, prefix_mask_action, suffix_mask, suffix_ar_mask
            )
            combined_positions = self._build_combined_positions(prefix_mask, prefix_mask_action, suffix_mask)

            (prefix_pre_logits, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=combined_mask,
                positions=combined_positions,
                adarms_cond=[None, adarms_cond],
            )
            action_loss = jnp.mean(
                jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t),
                axis=(-1, -2),
            )
            metrics = {"action_loss": jnp.mean(action_loss)}

            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                alignment_targets,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
                prefix_pre_logits,
                train=train,
                query_hidden_noise_rng=query_hidden_noise_rng,
            )
            return self._combine_oat_losses(
                action_loss,
                metrics,
                latent_loss=latent_loss,
                latent_metrics=latent_metrics,
                oat_recon_loss=oat_recon_loss,
                oat_recon_metrics=oat_recon_metrics,
            )

        if self.use_oat_pooled_cond and self._uses_query_hidden_pooled_cond(train=True):
            prefix_outputs = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
                adarms_cond=[None, None],
            )
            prefix_outs, kv_cache = prefix_outputs
            prefix_pre_logits = prefix_outs[0]
            query_memory = self._extract_query_memory_from_prefix_logits(prefix_pre_logits)
            oat_cond = self._encode_oat_cond_tokens(
                query_memory,
                self._get_pooled_cond_query_mask(actions.shape[0]),
            )
            oat_cond = oat_cond * pooled_cond_gate[:, None].astype(oat_cond.dtype)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, oat_cond=oat_cond
            )
            prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_action = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_action, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out_unused, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                cross_attn_memory=None,
                cross_attn_gate=jnp.ones((actions.shape[0],), dtype=bool),
            )
            assert prefix_out_unused is None
            action_loss = jnp.mean(
                jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t),
                axis=(-1, -2),
            )
            metrics = {"action_loss": jnp.mean(action_loss)}
            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                alignment_targets,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
                prefix_pre_logits,
                train=train,
                query_hidden_noise_rng=query_hidden_noise_rng,
            )
            return self._combine_oat_losses(
                action_loss,
                metrics,
                latent_loss=latent_loss,
                latent_metrics=latent_metrics,
                oat_recon_loss=oat_recon_loss,
                oat_recon_metrics=oat_recon_metrics,
            )

        cross_attn_memory = (
            self._build_projected_oat_expert_cross_attn_memory(oat_hidden_targets_for_expert, train=True)
            if oat_hidden_targets_for_expert is not None
            else None
        )
        cross_attn_gate = self._sample_cross_attn_gate(dropout_rng, actions.shape[0], train=train)
        latent_query_indices = self._get_latent_query_indices(prefix_mask, train=True)
        internal_query_memory_adapter_kwargs = self._get_internal_query_memory_adapter_kwargs(train=True)

        if self._uses_internal_query_hidden_expert_memory(train=True):
            prefix_outputs = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
                adarms_cond=[None, None],
                latent_query_indices=latent_query_indices,
                return_query_states=True,
                prefix_context_mask=prefix_context_mask,
                **internal_query_memory_adapter_kwargs,
            )
            prefix_outs, kv_cache, query_memory_states = prefix_outputs
            prefix_pre_logits = prefix_outs[0]
            prefix_context_summary = self._build_prefix_context_summary(prefix_pre_logits, prefix_context_mask)
            cross_attn_memory = self._resample_expert_memory(query_memory_states)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_action = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_action, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            cross_attn_memory_mask = self._build_expert_cross_attn_memory_mask(time, train=True)

            (prefix_out_unused, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                cross_attn_memory=cross_attn_memory,
                cross_attn_memory_mask=cross_attn_memory_mask,
                cross_attn_gate=cross_attn_gate,
                cross_attn_prefix_context=prefix_context_summary,
            )
            assert prefix_out_unused is None
            action_loss = jnp.mean(
                jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t),
                axis=(-1, -2),
            )
            metrics = {"action_loss": jnp.mean(action_loss)}

            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                alignment_targets,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
                prefix_pre_logits,
                train=train,
                query_hidden_noise_rng=query_hidden_noise_rng,
            )
            return self._combine_oat_losses(
                action_loss,
                metrics,
                latent_loss=latent_loss,
                latent_metrics=latent_metrics,
                oat_recon_loss=oat_recon_loss,
                oat_recon_metrics=oat_recon_metrics,
            )

        if (
            (cross_attn_memory is not None or not self.oat_latent_cross_attention_to_expert)
            and not self._uses_query_hidden_expert_memory(train=True)
            and not self._uses_mixed_expert_memory(train=True)
        ):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
            combined_mask = self._build_combined_attention_mask(
                prefix_attn_mask, prefix_mask_action, suffix_mask, suffix_ar_mask
            )
            combined_positions = self._build_combined_positions(prefix_mask, prefix_mask_action, suffix_mask)
            cross_attn_memory_mask = self._build_expert_cross_attn_memory_mask(time, train=True)

            (prefix_pre_logits, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=combined_mask,
                positions=combined_positions,
                adarms_cond=[None, adarms_cond],
                cross_attn_memory=cross_attn_memory,
                cross_attn_memory_mask=cross_attn_memory_mask,
                cross_attn_gate=cross_attn_gate,
                prefix_context_mask=prefix_context_mask,
            )
            action_loss = jnp.mean(
                jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t),
                axis=(-1, -2),
            )
            metrics = {"action_loss": jnp.mean(action_loss)}

            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                alignment_targets,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
                prefix_pre_logits,
                train=train,
                query_hidden_noise_rng=query_hidden_noise_rng,
            )
            return self._combine_oat_losses(
                action_loss,
                metrics,
                latent_loss=latent_loss,
                latent_metrics=latent_metrics,
                oat_recon_loss=oat_recon_loss,
                oat_recon_metrics=oat_recon_metrics,
            )

        prefix_outputs = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            adarms_cond=[None, None],
            prefix_context_mask=prefix_context_mask,
        )
        prefix_outs, kv_cache = prefix_outputs
        prefix_pre_logits = prefix_outs[0]
        prefix_context_summary = self._build_prefix_context_summary(prefix_pre_logits, prefix_context_mask)
        query_memory = (
            self._extract_query_expert_memory_from_prefix_logits(prefix_pre_logits, train=True)
            if self._uses_query_hidden_expert_memory(train=True)
            else None
        )

        if self._uses_mixed_expert_memory(train=True):
            query_cross_attn_memory = self._build_query_hidden_expert_cross_attn_memory(query_memory, train=True)
            cross_attn_memory = self._mix_expert_cross_attn_memory(
                cross_attn_memory,
                query_cross_attn_memory,
                expert_memory_mix_rng,
                actions.shape[0],
                train=True,
            )
        elif cross_attn_memory is None and self._uses_query_hidden_expert_memory(train=True):
            cross_attn_memory = self._build_query_hidden_expert_cross_attn_memory(query_memory, train=True)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_action = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_action, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        cross_attn_memory_mask = self._build_expert_cross_attn_memory_mask(time, train=True)

        (prefix_out_unused, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            cross_attn_memory=cross_attn_memory,
            cross_attn_memory_mask=cross_attn_memory_mask,
            cross_attn_gate=cross_attn_gate,
            cross_attn_prefix_context=prefix_context_summary,
        )
        assert prefix_out_unused is None
        action_loss = jnp.mean(
            jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t), axis=(-1, -2)
        )
        metrics = {"action_loss": jnp.mean(action_loss)}

        if self.use_oat_latent_alignment:
            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                alignment_targets,
                jnp.asarray(observation.oat_latent_mask, dtype=bool),
                prefix_pre_logits,
                train=train,
                query_hidden_noise_rng=query_hidden_noise_rng,
            )
            return self._combine_oat_losses(
                action_loss,
                metrics,
                latent_loss=latent_loss,
                latent_metrics=latent_metrics,
                oat_recon_loss=oat_recon_loss,
                oat_recon_metrics=oat_recon_metrics,
            )

        return self._combine_oat_losses(
            action_loss,
            metrics,
            oat_recon_loss=oat_recon_loss,
            oat_recon_metrics=oat_recon_metrics,
        )

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: OATObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        total_loss, _ = self._compute_loss_components(rng, observation, actions, train=train)
        return total_loss

    @override
    def compute_loss_and_metrics(
        self,
        rng: at.KeyArrayLike,
        observation: OATObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        return self._compute_loss_components(rng, observation, actions, train=train)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: OATObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_oat_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_attn_mask = self.embed_prefix(observation)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        use_prefix_context = self._uses_cross_attn_prefix_context()
        prefix_context_mask = self._build_prefix_context_mask(prefix_mask) if use_prefix_context else None
        latent_query_indices = self._get_latent_query_indices(prefix_mask, train=False)
        internal_query_memory_adapter_kwargs = self._get_internal_query_memory_adapter_kwargs(train=False)
        return_query_states = self._uses_internal_query_hidden_expert_memory(train=False)
        prefix_outputs = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None],
            latent_query_indices=latent_query_indices,
            return_query_states=return_query_states,
            prefix_context_mask=prefix_context_mask,
            **internal_query_memory_adapter_kwargs,
        )
        if return_query_states:
            prefix_outs, kv_cache, query_memory_states = prefix_outputs
        else:
            prefix_outs, kv_cache = prefix_outputs
            query_memory_states = None
        prefix_pre_logits = prefix_outs[0]
        prefix_context_summary = self._build_prefix_context_summary(prefix_pre_logits, prefix_context_mask)
        query_memory = (
            self._extract_query_memory_from_prefix_logits(prefix_pre_logits)
            if self._uses_query_hidden_pooled_cond(train=False)
            else None
        )
        query_expert_memory = (
            self._extract_query_expert_memory_from_prefix_logits(prefix_pre_logits, train=False)
            if self._uses_query_hidden_expert_memory(train=False) and not return_query_states
            else None
        )
        if return_query_states:
            cross_attn_memory = self._resample_expert_memory(query_memory_states)
        elif self._uses_query_hidden_expert_memory(train=False):
            cross_attn_memory = self._build_query_hidden_expert_cross_attn_memory(query_expert_memory, train=False)
        else:
            cross_attn_memory = None

        prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
        if self.use_oat_pooled_cond and self._uses_query_hidden_pooled_cond(train=False):
            pooled_query_cond = self._encode_oat_cond_tokens(
                query_memory,
                self._get_pooled_cond_query_mask(batch_size),
            )
        else:
            pooled_query_cond = None

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), oat_cond=pooled_query_cond
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_action = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_action, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            time_batch = jnp.broadcast_to(time, batch_size)
            cross_attn_memory_mask = self._build_expert_cross_attn_memory_mask(time_batch, train=False)

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                cross_attn_memory=cross_attn_memory,
                cross_attn_memory_mask=cross_attn_memory_mask,
                cross_attn_gate=jnp.ones((batch_size,), dtype=bool),
                cross_attn_prefix_context=prefix_context_summary,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
