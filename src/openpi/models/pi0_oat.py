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


class Pi0Oat(_model.BaseModel):
    def __init__(self, config: Pi0OatConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        if not config.pi05:
            raise ValueError("Pi0Oat currently only supports pi05=True.")

        self.pi05 = config.pi05
        self.oat_mode = str(config.oat_mode)
        self.use_oat_latent_alignment = bool(config.use_oat_latent_alignment)
        self.lambda_latent = float(config.lambda_latent)
        self.oat_num_queries = int(config.oat_num_queries or 0)
        self.oat_latent_dim = int(config.oat_latent_dim or 0)
        self.use_oat_alignment_target_stop_gradient = bool(config.use_oat_alignment_target_stop_gradient)
        self.use_oat_latent_reconstruction = bool(config.use_oat_latent_reconstruction)
        self.lambda_oat_recon = float(config.lambda_oat_recon)
        self.oat_alignment_target_space = "raw"
        self.oat_query_hidden_noise_std = float(config.oat_query_hidden_noise_std)
        self.oat_expert_memory_noise_std = float(config.oat_expert_memory_noise_std)
        self.oat_expert_memory_noise_train_prob = float(config.oat_expert_memory_noise_train_prob)

        paligemma_config = _gemma_oat.get_config(config.paligemma_variant)
        action_expert_config = _gemma_oat.get_config(config.action_expert_variant)
        self.paligemma_depth = paligemma_config.depth
        self.paligemma_width = paligemma_config.width
        llm = nnx_bridge.ToNNX(
            _gemma_oat.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                oat_mode=self.oat_mode,
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
            if self.oat_mode != "noexpert":
                self.query_memory_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
                self.query_memory_proj = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.query_oat_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.query_oat_proj = nnx.Linear(paligemma_config.width, self.oat_latent_dim, rngs=rngs)

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
        query_prediction = self._extract_query_raw_oat_from_prefix_logits(
            prefix_pre_logits, train=train, rng=query_hidden_noise_rng
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
        return self._extract_query_memory_from_prefix_logits(prefix_pre_logits, train=train, rng=rng)

    def _compute_loss_components(
        self,
        rng: at.KeyArrayLike,
        observation: OATObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng, expert_memory_noise_rng = jax.random.split(rng, 4)
        query_hidden_noise_rng = jax.random.fold_in(rng, 4)
        observation = preprocess_oat_observation(preprocess_rng, observation, train=train)
        if self.use_oat_latent_alignment and (observation.oat_latents is None or observation.oat_latent_mask is None):
            raise ValueError("use_oat_latent_alignment=True requires oat_latents and oat_latent_mask.")
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_attn_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask)
        combined_mask = self._build_combined_attention_mask(prefix_attn_mask, prefix_mask_action, suffix_mask, suffix_ar_mask)
        combined_positions = self._build_combined_positions(prefix_mask, prefix_mask_action, suffix_mask)
        raw_oat_targets = jnp.asarray(observation.oat_latents, dtype=jnp.float32) if self.use_oat_latent_alignment else None
        oat_hidden_targets = self._project_raw_oat_to_hidden(raw_oat_targets) if raw_oat_targets is not None else None
        cross_attn_memory = None
        cross_attn_memory_mask = None
        if self.oat_mode != "noexpert":
            raw_oat_memory = self._add_scale_aware_gaussian_noise_with_train_prob(
                raw_oat_targets, expert_memory_noise_rng, self.oat_expert_memory_noise_std,
                self.oat_expert_memory_noise_train_prob, train=train
            )
            cross_attn_memory = self._build_expert_cross_attn_memory(self._project_raw_oat_to_hidden(raw_oat_memory))
            cross_attn_memory_mask = jnp.asarray(observation.oat_latent_mask, dtype=bool)
        (prefix_pre_logits, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=combined_mask, positions=combined_positions,
            adarms_cond=[None, adarms_cond], cross_attn_memory=cross_attn_memory,
            cross_attn_memory_mask=cross_attn_memory_mask,
        )
        action_loss = jnp.mean(jnp.square(self.action_out_proj(suffix_out[:, -self.action_horizon :]) - u_t), axis=(-1, -2))
        metrics = {"action_loss": jnp.mean(action_loss)}
        latent_loss = latent_metrics = oat_recon_loss = oat_recon_metrics = None
        if self.use_oat_latent_alignment:
            latent_loss, latent_metrics = self._compute_latent_alignment_loss(
                raw_oat_targets, jnp.asarray(observation.oat_latent_mask, dtype=bool), prefix_pre_logits,
                train=train, query_hidden_noise_rng=query_hidden_noise_rng
            )
            if self.use_oat_latent_reconstruction:
                oat_recon_loss, oat_recon_metrics = self._compute_oat_reconstruction_loss(
                    oat_hidden_targets, observation.oat_latents, observation.oat_latent_mask
                )
        return self._combine_oat_losses(
            action_loss, metrics, latent_loss=latent_loss, latent_metrics=latent_metrics,
            oat_recon_loss=oat_recon_loss, oat_recon_metrics=oat_recon_metrics
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
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_attn_mask = self.embed_prefix(observation)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_outs, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
        )
        prefix_pre_logits = prefix_outs[0]
        cross_attn_memory = None
        cross_attn_memory_mask = None
        if self.oat_mode != "noexpert":
            query_hidden = self._extract_query_hidden_from_prefix_logits(prefix_pre_logits)
            cross_attn_memory = self._build_query_hidden_expert_cross_attn_memory(query_hidden, train=False)
            cross_attn_memory_mask = jnp.asarray(observation.oat_latent_mask, dtype=bool)
        prefix_mask_action = self._build_prefix_action_mask(prefix_mask)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            prefix_attn_mask_action = einops.repeat(prefix_mask_action, "b p -> b s p", s=suffix_tokens.shape[1])
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            full_attn_mask = jnp.concatenate([prefix_attn_mask_action, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask_action, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond], cross_attn_memory=cross_attn_memory,
                cross_attn_memory_mask=cross_attn_memory_mask,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0)))
        return x_0
