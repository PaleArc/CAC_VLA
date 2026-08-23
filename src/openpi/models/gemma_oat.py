# ruff: noqa: F821, FBT002, PLW0603, SLF001

# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as base_gemma
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = base_gemma.PALIGEMMA_VOCAB_SIZE
Config = base_gemma.Config
Variant = base_gemma.Variant
get_config = base_gemma.get_config
RMSNorm = base_gemma.RMSNorm
Embedder = base_gemma.Embedder
FeedForward = base_gemma.FeedForward

_HIDDEN_GATE_EVENT_COUNTER = 0


@at.typecheck
class Attention(nn.Module):
    configs: Sequence[Config]
    cache_dtype: str | None = None

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5
        k = _apply_rope(k, positions=positions)

        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            idx, cache_k, cache_v = kv_cache
            if xs[0] is not None:
                idx, k, v = _update_cache(k, v, idx, cache_k, cache_v, cache_dtype=self.cache_dtype)
            else:
                idx += k.shape[1]
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)
        else:
            idx, k, v = _init_cache(k, v, attn_mask.shape[-1], cache_dtype=self.cache_dtype)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        big_neg = -2.3819763e38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (idx, k, v)


@at.typecheck
class CrossAttention(nn.Module):
    query_config: Config
    memory_width: int

    @nn.compact
    def __call__(
        self,
        x: at.Float[at.Array, "b t d"],
        memory: at.Float[at.Array, "b s m"],
        memory_mask: at.Bool[at.Array, "b s"] | None = None,
        memory_logit_bias: at.Float[at.Array, "b s"] | None = None,
        debug_layer_index: at.Int[at.Array, ""] | None = None,
        debug_layer_enabled: at.Bool[at.Array, ""] | None = None,
    ) -> at.Float[at.Array, "b t d"]:
        dtype = x.dtype

        q_einsum = lora.Einsum(
            shape=(self.query_config.num_heads, self.query_config.width, self.query_config.head_dim),
            name="q_einsum",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            lora_config=self.query_config.lora_configs.get("attn"),
        )
        q = q_einsum("BTD,NDH->BTNH", x)
        kv_einsum = lora.Einsum(
            shape=(2, self.query_config.num_kv_heads, self.memory_width, self.query_config.head_dim),
            name="kv_einsum",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
            lora_config=self.query_config.lora_configs.get("attn"),
        )
        k, v = kv_einsum("BSD,2KDH->2BSKH", memory)

        q = q.astype(dtype)
        k = k.astype(dtype)
        v = v.astype(dtype)
        q *= self.query_config.head_dim**-0.5

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.query_config.num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        if memory_logit_bias is not None:
            logits = logits + memory_logit_bias[:, None, None, None, :].astype(logits.dtype)
        if memory_mask is not None:
            logits = jnp.where(
                memory_mask[:, None, None, None, :],
                logits,
                jnp.finfo(logits.dtype).min,
            )
        probs = jax.nn.softmax(logits, axis=-1).astype(dtype)
        _maybe_record_cross_attn_probs("latent_cross_attn", probs, debug_layer_index, debug_layer_enabled)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(self.query_config.num_heads, self.query_config.head_dim, self.query_config.width),
            name="attn_vec_einsum",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
            lora_config=self.query_config.lora_configs.get("attn"),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)


@at.typecheck
class Block(nn.Module):
    configs: tuple[Config, ...]
    latent_cross_attention_to_expert: bool = False
    use_cross_attn_residual_gate: bool = True
    use_cross_attn_hidden_gate: bool = False
    cross_attn_hidden_gate_bias_init: float = -5.0
    use_cross_attn_context_gate: bool = False
    cross_attn_context_gate_bias_init: float = -5.0
    cross_attn_context_gate_inputs: tuple[str, ...] = ("action", "attn", "prefix")
    use_cross_attn_kv_soft_gate: bool = False
    cross_attn_kv_soft_gate_bias_init: float = 0.0
    cache_dtype: str | None = None
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        latent_query_indices,
        cross_attn_memory,
        cross_attn_memory_mask,
        cross_attn_gate,
        cross_attn_layer_index,
        cross_attn_layer_enabled,
        prefix_context_mask,
        cross_attn_prefix_context,
        query_memory_norm_scale,
        query_memory_norm_bias,
        query_memory_norm_epsilon,
        query_memory_proj_kernel,
        query_memory_proj_bias,
        deterministic=True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn", cache_dtype=self.cache_dtype)

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        prefix_context_state = None
        if self.latent_cross_attention_to_expert:
            if xs[0] is not None and prefix_context_mask is not None and prefix_context_mask.shape[1] > 0:
                prefix_context_state = _masked_mean(xs[0], prefix_context_mask)
            elif cross_attn_prefix_context is not None and cross_attn_prefix_context.shape[-1] > 0:
                prefix_context_state = cross_attn_prefix_context

        query_state = None
        if self.latent_cross_attention_to_expert and xs[0] is not None and latent_query_indices is not None:
            query_state = jnp.take_along_axis(xs[0], latent_query_indices[..., None], axis=1)

            # 下面的意思是训练的时候使用VLM而不是用oat
            if query_memory_norm_scale is not None and query_memory_proj_kernel is not None:
                query_state = _apply_query_memory_adapter(
                    query_state,
                    query_memory_norm_scale,
                    query_memory_norm_bias,
                    query_memory_norm_epsilon,
                    query_memory_proj_kernel,
                    query_memory_proj_bias,
                )

        if self.latent_cross_attention_to_expert and len(xs) > 1 and xs[1] is not None:
            cross_attn_layer_scale = jnp.asarray(cross_attn_layer_enabled, dtype=xs[1].dtype)
            latent_memory = None
            latent_memory_mask = None
            if cross_attn_memory is not None and cross_attn_memory.shape[1] > 0:
                latent_memory = cross_attn_memory
                if cross_attn_memory_mask is not None and cross_attn_memory_mask.shape[1] > 0:
                    latent_memory_mask = cross_attn_memory_mask
            elif query_state is not None:
                latent_memory = query_state
                if cross_attn_memory_mask is not None and cross_attn_memory_mask.shape[1] > 0:
                    latent_memory_mask = cross_attn_memory_mask

            if latent_memory is not None and latent_memory.shape[1] > 0:
                normed_action_x, latent_gate = RMSNorm(name=_name("pre_latent_cross_attention_norm", 1))(
                    xs[1], adarms_cond[1]
                )
                memory_logit_bias = None
                if self.use_cross_attn_kv_soft_gate:
                    if prefix_context_state is None:
                        prefix_context_state = jnp.zeros(
                            (normed_action_x.shape[0], self.configs[0].width),
                            dtype=normed_action_x.dtype,
                        )
                    action_summary = jnp.mean(normed_action_x, axis=1)
                    action_gate_state = nn.Dense(
                        self.configs[0].width,
                        name=_name("latent_cross_attn_kv_gate_action_proj", 1),
                    )(action_summary)
                    memory_gate_state = nn.Dense(
                        self.configs[0].width,
                        name=_name("latent_cross_attn_kv_gate_memory_proj", 1),
                    )(latent_memory)
                    prefix_gate_state = nn.Dense(
                        self.configs[0].width,
                        name=_name("latent_cross_attn_kv_gate_prefix_proj", 1),
                    )(prefix_context_state)
                    kv_gate_input = jnp.tanh(
                        memory_gate_state + action_gate_state[:, None, :] + prefix_gate_state[:, None, :]
                    )
                    kv_gate_score = nn.Dense(
                        1,
                        kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.constant(self.cross_attn_kv_soft_gate_bias_init),
                        name=_name("latent_cross_attn_kv_gate_score", 1),
                    )(kv_gate_input)
                    kv_gate = jax.nn.sigmoid(kv_gate_score)
                    memory_logit_bias = jnp.log(kv_gate[..., 0].astype(jnp.float32) + jnp.finfo(jnp.float32).eps)
                latent_attn = CrossAttention(
                    query_config=self.configs[1],
                    memory_width=self.configs[0].width,
                    name=_name("latent_cross_attn", 1),
                )(
                    normed_action_x,
                    latent_memory,
                    latent_memory_mask,
                    memory_logit_bias,
                    cross_attn_layer_index,
                    cross_attn_layer_enabled,
                )
                latent_attn = latent_attn * cross_attn_gate[:, None, None].astype(latent_attn.dtype)
                hidden_gate_record_name = None
                if self.use_cross_attn_residual_gate and self.use_cross_attn_hidden_gate:
                    if self.use_cross_attn_context_gate:
                        context_gate_state = None
                        if "action" in self.cross_attn_context_gate_inputs:
                            action_summary = jnp.mean(normed_action_x, axis=1)
                            context_gate_state = nn.Dense(
                                self.configs[1].width,
                                name=_name("latent_cross_attn_context_gate_action_proj", 1),
                            )(action_summary)
                        if "attn" in self.cross_attn_context_gate_inputs:
                            attn_summary = jnp.mean(latent_attn, axis=1)
                            attn_gate_state = nn.Dense(
                                self.configs[1].width,
                                name=_name("latent_cross_attn_context_gate_attn_proj", 1),
                            )(attn_summary)
                            context_gate_state = (
                                attn_gate_state if context_gate_state is None else context_gate_state + attn_gate_state
                            )
                        if "prefix" in self.cross_attn_context_gate_inputs:
                            if prefix_context_state is None:
                                prefix_context_state = jnp.zeros(
                                    (normed_action_x.shape[0], self.configs[0].width),
                                    dtype=normed_action_x.dtype,
                                )
                            prefix_gate_state = nn.Dense(
                                self.configs[1].width,
                                name=_name("latent_cross_attn_context_gate_prefix_proj", 1),
                            )(prefix_context_state)
                            context_gate_state = (
                                prefix_gate_state
                                if context_gate_state is None
                                else context_gate_state + prefix_gate_state
                            )
                        if context_gate_state is None:
                            raise ValueError("cross_attn_context_gate_inputs must be non-empty.")
                        context_gate_state = jnp.tanh(context_gate_state)
                        context_gate = nn.Dense(
                            self.configs[1].width,
                            kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.constant(self.cross_attn_context_gate_bias_init),
                            name=_name("latent_cross_attn_context_gate_score", 1),
                        )(context_gate_state)
                        latent_gate = jax.nn.sigmoid(context_gate)[:, None, :].astype(latent_attn.dtype)
                        hidden_gate_record_name = _name("latent_cross_attn_context_gate", 1)
                    else:
                        hidden_gate = nn.Dense(
                            self.configs[1].width,
                            kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.constant(self.cross_attn_hidden_gate_bias_init),
                            name=_name("latent_cross_attn_hidden_gate", 1),
                        )(normed_action_x)
                        latent_gate = jax.nn.sigmoid(hidden_gate).astype(latent_attn.dtype)
                        hidden_gate_record_name = _name("latent_cross_attn_hidden_gate", 1)
                latent_attn = drop(latent_attn, deterministic)
                # cross_attn_layer_scale 是按层来开cross atttn用到
                latent_attn = latent_attn * cross_attn_layer_scale
                if self.use_cross_attn_residual_gate:
                    latent_gate = latent_gate * cross_attn_layer_scale.astype(latent_gate.dtype)
                    if hidden_gate_record_name is not None:
                        _maybe_record_hidden_gate(
                            hidden_gate_record_name,
                            latent_gate,
                            latent_attn,
                            cross_attn_layer_index,
                            cross_attn_layer_enabled,
                            xs[1],
                        )
                    xs[1] = _gated_residual(xs[1], latent_attn, latent_gate)
                else:
                    xs[1] = xs[1] + latent_attn
                xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        if query_state is None:
            batch_size = next(x.shape[0] for x in xs if x is not None)
            query_state = jnp.zeros(
                (batch_size, 0, self.configs[0].width), dtype=xs[0].dtype if xs[0] is not None else jnp.float32
            )

        return xs, (kv_cache, query_state)


KVCache: TypeAlias = tuple[
    at.Int[at.Array, "l b"],
    at.Float[at.Array, "l b _t _k _h"],
    at.Float[at.Array, "l b _t _v _h"],
]


@at.typecheck
class Module(nn.Module):
    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    latent_cross_attention_to_expert: bool = False
    use_cross_attn_residual_gate: bool = True
    use_cross_attn_hidden_gate: bool = False
    cross_attn_hidden_gate_bias_init: float = -5.0
    use_cross_attn_context_gate: bool = False
    cross_attn_context_gate_bias_init: float = -5.0
    cross_attn_context_gate_inputs: tuple[str, ...] = ("action", "attn", "prefix")
    use_cross_attn_kv_soft_gate: bool = False
    cross_attn_kv_soft_gate_bias_init: float = 0.0
    cross_attn_last_n_layers: int | None = None
    cache_dtype: str | None = None

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,
                nn.broadcast,
                nn.broadcast,
                0,
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            latent_cross_attention_to_expert=self.latent_cross_attention_to_expert,
            use_cross_attn_residual_gate=self.use_cross_attn_residual_gate,
            use_cross_attn_hidden_gate=self.use_cross_attn_hidden_gate,
            cross_attn_hidden_gate_bias_init=self.cross_attn_hidden_gate_bias_init,
            use_cross_attn_context_gate=self.use_cross_attn_context_gate,
            cross_attn_context_gate_bias_init=self.cross_attn_context_gate_bias_init,
            cross_attn_context_gate_inputs=self.cross_attn_context_gate_inputs,
            use_cross_attn_kv_soft_gate=self.use_cross_attn_kv_soft_gate,
            cross_attn_kv_soft_gate_bias_init=self.cross_attn_kv_soft_gate_bias_init,
            cache_dtype=self.cache_dtype,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
        latent_query_indices: at.Int[at.Array, "b q"] | None = None,
        cross_attn_memory: at.Float[at.Array, "l b q d"] | None = None,
        cross_attn_memory_mask: at.Bool[at.Array, "b q"] | None = None,
        cross_attn_gate: at.Bool[at.Array, "b"] | None = None,
        prefix_context_mask: at.Bool[at.Array, "b p"] | None = None,
        cross_attn_prefix_context: at.Float[at.Array, "b d"] | None = None,
        query_memory_norm_scale: at.Float[at.Array, "d"] | None = None,
        query_memory_norm_bias: at.Float[at.Array, "d"] | None = None,
        query_memory_norm_epsilon: float = 1e-6,
        query_memory_proj_kernel: at.Float[at.Array, "d d"] | None = None,
        query_memory_proj_bias: at.Float[at.Array, "d"] | None = None,
        return_query_states: bool = False,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        batch_size = positions.shape[0]
        if latent_query_indices is None:
            latent_query_indices = jnp.zeros((batch_size, 0), dtype=jnp.int32)
        if cross_attn_memory is None:
            cross_attn_memory = jnp.zeros(
                (self.configs[0].depth, batch_size, 0, self.configs[0].width),
                dtype=jnp.dtype(self.embed_dtype),
            )
        if cross_attn_memory_mask is None:
            cross_attn_memory_mask = jnp.zeros((batch_size, 0), dtype=bool)
        if cross_attn_gate is None:
            cross_attn_gate = jnp.ones((batch_size,), dtype=bool)
        if prefix_context_mask is None:
            prefix_context_mask = jnp.zeros((batch_size, 0), dtype=bool)
        if cross_attn_prefix_context is None:
            cross_attn_prefix_context = jnp.zeros(
                (batch_size, self.configs[0].width),
                dtype=jnp.dtype(self.embed_dtype),
            )
        cross_attn_layer_index = jnp.arange(self.configs[0].depth, dtype=jnp.int32)
        if self.cross_attn_last_n_layers is None:
            cross_attn_layer_enabled = jnp.ones((self.configs[0].depth,), dtype=bool)
        else:
            first_enabled_layer = self.configs[0].depth - self.cross_attn_last_n_layers
            cross_attn_layer_enabled = cross_attn_layer_index >= first_enabled_layer

        embedded, (kv_cache, query_states) = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            latent_query_indices,
            cross_attn_memory.astype(self.embed_dtype),
            cross_attn_memory_mask,
            cross_attn_gate,
            cross_attn_layer_index,
            cross_attn_layer_enabled,
            prefix_context_mask,
            cross_attn_prefix_context.astype(self.embed_dtype),
            query_memory_norm_scale,
            query_memory_norm_bias,
            query_memory_norm_epsilon,
            query_memory_proj_kernel,
            query_memory_proj_bias,
            deterministic,
        )

        out = [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if return_query_states:
            return out, kv_cache, query_states
        return out, kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        latent_query_indices = (
            jnp.zeros((1, 1), dtype=jnp.int32)
            if self.latent_cross_attention_to_expert and len(self.configs) > 1
            else None
        )
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            latent_query_indices=latent_query_indices,
        )


def _masked_mean(x: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"]) -> at.Float[at.Array, "b d"]:
    mask = jnp.asarray(mask, dtype=bool)
    weights = mask[..., None].astype(x.dtype)
    denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
    return jnp.sum(x * weights, axis=1) / denom


def _apply_rope(x, *, positions, max_wavelength=10_000):
    return base_gemma._apply_rope(x, positions=positions, max_wavelength=max_wavelength)


def _name(name, i):
    return base_gemma._name(name, i)


def _maybe_record_hidden_gate(
    name: str,
    gate: at.Float[at.Array, "b t d"],
    residual: at.Float[at.Array, "b t d"] | None = None,
    layer_index=None,
    layer_enabled=None,
    hidden_state: at.Float[at.Array, "b t d"] | None = None,
) -> None:
    if not os.environ.get("OPENPI_OAT_HIDDEN_GATE_DUMP_DIR"):
        return
    if layer_index is None or layer_enabled is None:
        if residual is None:
            jax.debug.callback(lambda x: _record_hidden_gate(name, x), gate)
        elif hidden_state is None:
            jax.debug.callback(lambda x, y: _record_hidden_gate(name, x, residual=y), gate, residual)
        else:
            jax.debug.callback(
                lambda x, y, h: _record_hidden_gate(name, x, residual=y, hidden_state=h),
                gate,
                residual,
                hidden_state,
            )
    elif residual is None:
        jax.debug.callback(lambda x, i, e: _record_hidden_gate(name, x, i, e), gate, layer_index, layer_enabled)
    elif hidden_state is None:
        jax.debug.callback(
            lambda x, y, i, e: _record_hidden_gate(name, x, i, e, y),
            gate,
            residual,
            layer_index,
            layer_enabled,
        )
    else:
        jax.debug.callback(
            lambda x, y, h, i, e: _record_hidden_gate(name, x, i, e, y, h),
            gate,
            residual,
            hidden_state,
            layer_index,
            layer_enabled,
        )


def _maybe_record_cross_attn_probs(
    name: str,
    probs: at.Float[at.Array, "b k g t s"],
    layer_index=None,
    layer_enabled=None,
) -> None:
    if not os.environ.get("OPENPI_OAT_HIDDEN_GATE_DUMP_DIR"):
        return
    if layer_index is None or layer_enabled is None:
        jax.debug.callback(lambda x: _record_cross_attn_probs(name, x), probs)
    else:
        jax.debug.callback(lambda x, i, e: _record_cross_attn_probs(name, x, i, e), probs, layer_index, layer_enabled)


def _record_hidden_gate(
    name: str, gate, layer_index=None, layer_enabled=None, residual=None, hidden_state=None
) -> None:
    dump_dir = os.environ.get("OPENPI_OAT_HIDDEN_GATE_DUMP_DIR")
    if not dump_dir:
        return

    global _HIDDEN_GATE_EVENT_COUNTER
    event_index = _HIDDEN_GATE_EVENT_COUNTER
    _HIDDEN_GATE_EVENT_COUNTER += 1

    gate_np = np.asarray(gate, dtype=np.float32)
    context = os.environ.get("OPENPI_OAT_HIDDEN_GATE_CONTEXT", "unknown")
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "event_index": event_index,
        "context": context,
        "name": name,
        "shape": list(gate_np.shape),
        "mean": float(gate_np.mean()),
        "std": float(gate_np.std()),
        "min": float(gate_np.min()),
        "max": float(gate_np.max()),
        "p_gt_001": float(np.mean(gate_np > 0.01)),
        "p_gt_005": float(np.mean(gate_np > 0.05)),
        "p_gt_010": float(np.mean(gate_np > 0.10)),
        "p_gt_050": float(np.mean(gate_np > 0.50)),
        "per_token_mean": np.mean(gate_np, axis=-1).tolist(),
        "per_token_max": np.max(gate_np, axis=-1).tolist(),
    }
    if layer_index is not None:
        stats["layer_index"] = int(np.asarray(layer_index))
    if layer_enabled is not None:
        stats["layer_enabled"] = bool(np.asarray(layer_enabled))
    if residual is not None:
        residual_np = np.asarray(residual, dtype=np.float32)
        gated_residual_np = gate_np * residual_np
        eps = np.finfo(np.float32).eps
        stats["effective_injection_ratio"] = float(
            np.linalg.norm(gated_residual_np) / (np.linalg.norm(residual_np) + eps)
        )
        stats["residual_norm"] = float(np.linalg.norm(residual_np))
        stats["gated_residual_norm"] = float(np.linalg.norm(gated_residual_np))
        if hidden_state is not None:
            hidden_state_np = np.asarray(hidden_state, dtype=np.float32)
            stats["hidden_state_norm"] = float(np.linalg.norm(hidden_state_np))
            stats["latent_conditioning_strength"] = float(
                np.linalg.norm(gated_residual_np) / (np.linalg.norm(hidden_state_np) + eps)
            )

    with (out_dir / "hidden_gate_stats.jsonl").open("a") as f:
        f.write(json.dumps(stats) + "\n")

    if os.environ.get("OPENPI_OAT_HIDDEN_GATE_SAVE_ARRAYS") == "1":
        safe_context = "".join(c if c.isalnum() or c in "._-" else "_" for c in context)
        np.save(out_dir / f"{event_index:06d}_{safe_context}_{name}.npy", gate_np)


def _record_cross_attn_probs(name: str, probs, layer_index=None, layer_enabled=None) -> None:
    dump_dir = os.environ.get("OPENPI_OAT_HIDDEN_GATE_DUMP_DIR")
    if not dump_dir:
        return

    global _HIDDEN_GATE_EVENT_COUNTER
    event_index = _HIDDEN_GATE_EVENT_COUNTER
    _HIDDEN_GATE_EVENT_COUNTER += 1

    probs_np = np.asarray(probs, dtype=np.float32)
    context = os.environ.get("OPENPI_OAT_HIDDEN_GATE_CONTEXT", "unknown")
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # probs: B x K x G x T x S. Average heads/groups to show action-token -> memory-token selection.
    token_memory = probs_np.mean(axis=(1, 2))
    per_memory_mean = token_memory.mean(axis=1)
    top_indices = np.argsort(-token_memory, axis=-1)[..., : min(5, token_memory.shape[-1])]
    top_values = np.take_along_axis(token_memory, top_indices, axis=-1)

    stats = {
        "event_index": event_index,
        "context": context,
        "name": name,
        "shape": list(probs_np.shape),
        "token_memory_shape": list(token_memory.shape),
        "entropy_mean": float(-(token_memory * np.log(np.clip(token_memory, 1e-9, 1.0))).sum(axis=-1).mean()),
        "per_memory_mean": per_memory_mean.tolist(),
        "per_token_top_memory_indices": top_indices.tolist(),
        "per_token_top_memory_values": top_values.tolist(),
    }
    if layer_index is not None:
        stats["layer_index"] = int(np.asarray(layer_index))
    if layer_enabled is not None:
        stats["layer_enabled"] = bool(np.asarray(layer_enabled))

    with (out_dir / "cross_attn_probs_stats.jsonl").open("a") as f:
        f.write(json.dumps(stats) + "\n")

    if os.environ.get("OPENPI_OAT_HIDDEN_GATE_SAVE_ARRAYS") == "1":
        safe_context = "".join(c if c.isalnum() or c in "._-" else "_" for c in context)
        np.save(out_dir / f"{event_index:06d}_{safe_context}_{name}_token_memory.npy", token_memory)


def _gated_residual(x, y, gate):
    return base_gemma._gated_residual(x, y, gate)


def _init_cache(k, v, cache_size, cache_dtype=None):
    prefill_len = k.shape[1]
    pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
    cache_dtype = cache_dtype or k.dtype
    k_cache = jnp.pad(k.astype(cache_dtype), pad_width)
    v_cache = jnp.pad(v.astype(cache_dtype), pad_width)
    idx = jnp.zeros((k.shape[0],), dtype=jnp.int32) + prefill_len
    return idx, k_cache, v_cache


def _update_cache(k, v, idx, k_cache, v_cache, cache_dtype=None):
    assert k.shape[1] == 1, "Only support kv-cache updates of length 1"
    cache_dtype = cache_dtype or k.dtype
    indices = (0, idx[0], 0, 0)
    k_new = jax.lax.dynamic_update_slice(k_cache, k.astype(cache_dtype), indices)
    v_new = jax.lax.dynamic_update_slice(v_cache, v.astype(cache_dtype), indices)
    idx_new = idx + 1
    return idx_new, k_new, v_new


def _apply_query_memory_adapter(
    query_state,
    norm_scale,
    norm_bias,
    norm_epsilon,
    proj_kernel,
    proj_bias,
):
    x = jnp.asarray(query_state, dtype=jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    x = centered * jax.lax.rsqrt(var + norm_epsilon)
    if norm_scale is not None:
        x = x * jnp.asarray(norm_scale, dtype=x.dtype)
    if norm_bias is not None:
        x = x + jnp.asarray(norm_bias, dtype=x.dtype)
    x = jnp.einsum("bqd,df->bqf", x, jnp.asarray(proj_kernel, dtype=x.dtype), preferred_element_type=jnp.float32)
    if proj_bias is not None:
        x = x + jnp.asarray(proj_bias, dtype=x.dtype)
    return x.astype(query_state.dtype)
