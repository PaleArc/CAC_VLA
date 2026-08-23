import dataclasses
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)
OATCrossAttnContextGateInput: TypeAlias = Literal["action", "attn", "prefix"]


@at.typecheck
@struct.dataclass
class OATObservation(_model.Observation[ArrayT], Generic[ArrayT]):
    oat_latents: at.Float[ArrayT, "*b q d"] | None = None
    oat_latent_mask: at.Bool[ArrayT, "*b q"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "OATObservation[ArrayT]":
        base = _model.Observation.from_dict(data)
        base_dict = dataclasses.asdict(base)
        return cls(
            **base_dict,
            oat_latents=data.get("oat_latents"),
            oat_latent_mask=data.get("oat_latent_mask"),
        )


def preprocess_oat_observation(
    rng: at.KeyArrayLike | None,
    observation: _model.Observation | OATObservation,
    *,
    train: bool = False,
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> OATObservation:
    base_obs = _model.preprocess_observation(
        rng,
        observation,
        train=train,
        image_keys=image_keys,
        image_resolution=image_resolution,
    )
    return OATObservation(
        images=base_obs.images,
        image_masks=base_obs.image_masks,
        state=base_obs.state,
        tokenized_prompt=base_obs.tokenized_prompt,
        tokenized_prompt_mask=base_obs.tokenized_prompt_mask,
        token_ar_mask=base_obs.token_ar_mask,
        token_loss_mask=base_obs.token_loss_mask,
        oat_latents=getattr(observation, "oat_latents", None),
        oat_latent_mask=getattr(observation, "oat_latent_mask", None),
    )


if TYPE_CHECKING:
    from openpi.models.pi0_oat import Pi0Oat


@dataclasses.dataclass(frozen=True)
class Pi0OatConfig(pi0_config.Pi0Config):
    use_oat_latent_alignment: bool = False
    lambda_latent: float = 0.0
    oat_num_queries: int | None = None
    oat_latent_dim: int | None = None
    use_oat_alignment_target_stop_gradient: bool = False
    use_oat_latent_reconstruction: bool = False
    lambda_oat_recon: float = 0.0
    use_oat_pooled_cond: bool = False
    oat_pooled_cond_source_train: Literal["projected_oat", "query_hidden"] = "projected_oat"
    oat_pooled_cond_source_infer: Literal["none", "query_hidden"] = "none"
    oat_pooled_cond_dropout_rate: float = 0.0
    oat_queries_visible_to_action_expert: bool = True
    oat_latent_cross_attention_to_expert: bool = False
    oat_expert_memory_source_train: Literal["query_hidden", "projected_oat", "mixed"] = "query_hidden"
    oat_expert_memory_source_infer: Literal["none", "query_hidden"] = "query_hidden"
    oat_expert_memory_projected_oat_train_prob: float = 1.0
    oat_expert_memory_dropout_rate: float = 0.1
    oat_expert_memory_time_schedule: Literal["none", "ceil_linear", "ceil_sin", "staged_4_6_8"] = "none"
    oat_expert_memory_schedule_domain: Literal["t", "log_snr"] = "t"
    oat_expert_memory_min_tokens: int = 1
    oat_expert_memory_token_shift: int = 3
    use_oat_expert_memory_query_resampler: bool = False
    oat_expert_memory_num_queries: int = 4
    use_internal_query_memory_for_expert_attn: bool = False
    use_oat_cross_attn_residual_gate: bool = True
    use_oat_cross_attn_hidden_gate: bool = False
    oat_cross_attn_hidden_gate_bias_init: float = -5.0
    use_oat_cross_attn_context_gate: bool = False
    oat_cross_attn_context_gate_bias_init: float = -5.0
    oat_cross_attn_context_gate_inputs: tuple[OATCrossAttnContextGateInput, ...] = ("action", "attn", "prefix")
    use_oat_cross_attn_kv_soft_gate: bool = False
    oat_cross_attn_kv_soft_gate_bias_init: float = 0.0
    oat_cross_attn_last_n_layers: int | None = None
    oat_alignment_target_space: Literal["hidden", "raw"] = "hidden"
    oat_expert_memory_input_space: Literal["hidden", "raw"] = "hidden"
    oat_query_hidden_noise_std: float = 0.0
    oat_expert_memory_noise_std: float = 0.0
    oat_expert_memory_noise_train_prob: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0OatConfig currently only supports pi05=True.")
        if not self.use_oat_latent_alignment:
            object.__setattr__(self, "lambda_latent", 0.0)
            object.__setattr__(self, "use_oat_latent_reconstruction", False)
            object.__setattr__(self, "lambda_oat_recon", 0.0)
        elif self.oat_num_queries is None or self.oat_latent_dim is None:
            raise ValueError("oat_num_queries and oat_latent_dim are required when use_oat_latent_alignment=True.")
        if self.use_oat_latent_reconstruction and not self.use_oat_latent_alignment:
            raise ValueError("use_oat_latent_reconstruction=True requires use_oat_latent_alignment=True.")
        if self.lambda_oat_recon < 0.0:
            raise ValueError("lambda_oat_recon must be >= 0.0.")
        if not self.use_oat_latent_reconstruction:
            object.__setattr__(self, "lambda_oat_recon", 0.0)
        if self.use_oat_pooled_cond and not self.use_oat_latent_alignment:
            raise ValueError("use_oat_pooled_cond=True requires use_oat_latent_alignment=True.")
        if self.oat_latent_cross_attention_to_expert:
            if not self.use_oat_latent_alignment:
                raise ValueError("oat_latent_cross_attention_to_expert=True requires use_oat_latent_alignment=True.")
            if self.oat_queries_visible_to_action_expert:
                raise ValueError(
                    "oat_latent_cross_attention_to_expert=True requires oat_queries_visible_to_action_expert=False."
                )
        valid_pooled_train_sources = {"projected_oat", "query_hidden"}
        if self.oat_pooled_cond_source_train not in valid_pooled_train_sources:
            raise ValueError(
                "oat_pooled_cond_source_train must be one of "
                f"{sorted(valid_pooled_train_sources)}, got {self.oat_pooled_cond_source_train!r}."
            )
        valid_pooled_infer_sources = {"none", "query_hidden"}
        if self.oat_pooled_cond_source_infer not in valid_pooled_infer_sources:
            raise ValueError(
                "oat_pooled_cond_source_infer must be one of "
                f"{sorted(valid_pooled_infer_sources)}, got {self.oat_pooled_cond_source_infer!r}."
            )
        valid_train_sources = {"query_hidden", "projected_oat", "mixed"}
        if self.oat_expert_memory_source_train not in valid_train_sources:
            raise ValueError(
                "oat_expert_memory_source_train must be one of "
                f"{sorted(valid_train_sources)}, got {self.oat_expert_memory_source_train!r}."
            )
        valid_infer_sources = {"none", "query_hidden"}
        if self.oat_expert_memory_source_infer not in valid_infer_sources:
            raise ValueError(
                "oat_expert_memory_source_infer must be one of "
                f"{sorted(valid_infer_sources)}, got {self.oat_expert_memory_source_infer!r}."
            )
        if not 0.0 <= self.oat_pooled_cond_dropout_rate < 1.0:
            raise ValueError("oat_pooled_cond_dropout_rate must be in [0.0, 1.0).")
        if not 0.0 <= self.oat_expert_memory_projected_oat_train_prob <= 1.0:
            raise ValueError("oat_expert_memory_projected_oat_train_prob must be in [0.0, 1.0].")
        if not 0.0 <= self.oat_expert_memory_dropout_rate < 1.0:
            raise ValueError("oat_expert_memory_dropout_rate must be in [0.0, 1.0).")
        valid_time_schedules = {"none", "ceil_linear", "ceil_sin", "staged_4_6_8"}
        if self.oat_expert_memory_time_schedule not in valid_time_schedules:
            raise ValueError(
                "oat_expert_memory_time_schedule must be one of "
                f"{sorted(valid_time_schedules)}, got {self.oat_expert_memory_time_schedule!r}."
            )
        valid_schedule_domains = {"t", "log_snr"}
        if self.oat_expert_memory_schedule_domain not in valid_schedule_domains:
            raise ValueError(
                "oat_expert_memory_schedule_domain must be one of "
                f"{sorted(valid_schedule_domains)}, got {self.oat_expert_memory_schedule_domain!r}."
            )
        if self.oat_expert_memory_min_tokens < 1:
            raise ValueError("oat_expert_memory_min_tokens must be >= 1.")
        if self.oat_expert_memory_token_shift < 0:
            raise ValueError("oat_expert_memory_token_shift must be >= 0.")
        if self.oat_expert_memory_num_queries < 1:
            raise ValueError("oat_expert_memory_num_queries must be >= 1.")
        valid_alignment_spaces = {"hidden", "raw"}
        if self.oat_alignment_target_space not in valid_alignment_spaces:
            raise ValueError(
                "oat_alignment_target_space must be one of "
                f"{sorted(valid_alignment_spaces)}, got {self.oat_alignment_target_space!r}."
            )
        valid_memory_spaces = {"hidden", "raw"}
        if self.oat_expert_memory_input_space not in valid_memory_spaces:
            raise ValueError(
                "oat_expert_memory_input_space must be one of "
                f"{sorted(valid_memory_spaces)}, got {self.oat_expert_memory_input_space!r}."
            )
        if self.oat_query_hidden_noise_std < 0.0:
            raise ValueError("oat_query_hidden_noise_std must be >= 0.0.")
        if self.oat_expert_memory_noise_std < 0.0:
            raise ValueError("oat_expert_memory_noise_std must be >= 0.0.")
        if not 0.0 <= self.oat_expert_memory_noise_train_prob <= 1.0:
            raise ValueError("oat_expert_memory_noise_train_prob must be in [0.0, 1.0].")
        if not self.use_oat_pooled_cond:
            object.__setattr__(self, "oat_pooled_cond_dropout_rate", 0.0)
        if not self.use_oat_latent_alignment:
            object.__setattr__(self, "oat_query_hidden_noise_std", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_std", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_train_prob", 0.0)
            object.__setattr__(self, "oat_alignment_target_space", "hidden")
            object.__setattr__(self, "oat_expert_memory_input_space", "hidden")
        if not self.oat_latent_cross_attention_to_expert:
            object.__setattr__(self, "oat_expert_memory_dropout_rate", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_std", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_train_prob", 0.0)
            object.__setattr__(self, "oat_expert_memory_input_space", "hidden")
            object.__setattr__(self, "oat_expert_memory_time_schedule", "none")
            object.__setattr__(self, "oat_expert_memory_schedule_domain", "t")
            object.__setattr__(self, "use_oat_expert_memory_query_resampler", False)
            object.__setattr__(self, "use_oat_cross_attn_residual_gate", True)
            object.__setattr__(self, "use_oat_cross_attn_hidden_gate", False)
            object.__setattr__(self, "use_oat_cross_attn_context_gate", False)
            object.__setattr__(self, "use_oat_cross_attn_kv_soft_gate", False)
            object.__setattr__(self, "oat_cross_attn_last_n_layers", None)
        if not self.use_oat_cross_attn_residual_gate or not self.use_oat_cross_attn_hidden_gate:
            object.__setattr__(self, "use_oat_cross_attn_context_gate", False)
        object.__setattr__(self, "oat_cross_attn_context_gate_inputs", tuple(self.oat_cross_attn_context_gate_inputs))
        if self.use_oat_cross_attn_context_gate:
            valid_context_gate_inputs = {"action", "attn", "prefix"}
            if not self.oat_cross_attn_context_gate_inputs:
                raise ValueError("oat_cross_attn_context_gate_inputs must be non-empty when context gate is enabled.")
            invalid_context_gate_inputs = set(self.oat_cross_attn_context_gate_inputs) - valid_context_gate_inputs
            if invalid_context_gate_inputs:
                raise ValueError(
                    "oat_cross_attn_context_gate_inputs must only contain "
                    f"{sorted(valid_context_gate_inputs)}, got {sorted(invalid_context_gate_inputs)}."
                )
        if self.oat_cross_attn_last_n_layers is not None and self.oat_cross_attn_last_n_layers <= 0:
            raise ValueError("oat_cross_attn_last_n_layers must be a positive integer or None.")
        if (
            self.oat_expert_memory_source_train in {"projected_oat", "mixed"}
            and not self.oat_latent_cross_attention_to_expert
        ):
            raise ValueError(
                "oat_expert_memory_source_train='projected_oat' or 'mixed' requires "
                "oat_latent_cross_attention_to_expert=True."
            )
        if self.oat_expert_memory_source_train in {"projected_oat", "mixed"} and not self.use_oat_latent_alignment:
            raise ValueError(
                "oat_expert_memory_source_train='projected_oat' or 'mixed' requires use_oat_latent_alignment=True."
            )
        if self.oat_expert_memory_input_space == "raw" and self.oat_expert_memory_source_train == "mixed":
            raise ValueError("oat_expert_memory_input_space='raw' does not support mixed expert memory.")
        if self.oat_expert_memory_input_space == "raw" and self.use_internal_query_memory_for_expert_attn:
            raise ValueError(
                "oat_expert_memory_input_space='raw' does not support use_internal_query_memory_for_expert_attn=True."
            )
        if self.use_internal_query_memory_for_expert_attn:
            if not self.oat_latent_cross_attention_to_expert:
                raise ValueError(
                    "use_internal_query_memory_for_expert_attn=True requires oat_latent_cross_attention_to_expert=True."
                )
            if self.oat_expert_memory_source_train != "query_hidden":
                raise ValueError(
                    "use_internal_query_memory_for_expert_attn=True requires "
                    "oat_expert_memory_source_train='query_hidden'."
                )
            if self.oat_expert_memory_source_infer != "query_hidden":
                raise ValueError(
                    "use_internal_query_memory_for_expert_attn=True requires "
                    "oat_expert_memory_source_infer='query_hidden'."
                )
        if self.oat_expert_memory_time_schedule != "none":
            if not self.oat_latent_cross_attention_to_expert:
                raise ValueError(
                    "oat_expert_memory_time_schedule!='none' requires oat_latent_cross_attention_to_expert=True."
                )
            if not self.use_oat_latent_alignment:
                raise ValueError("oat_expert_memory_time_schedule!='none' requires use_oat_latent_alignment=True.")
        if self.use_oat_expert_memory_query_resampler and not self.oat_latent_cross_attention_to_expert:
            raise ValueError(
                "use_oat_expert_memory_query_resampler=True requires oat_latent_cross_attention_to_expert=True."
            )
        if self.use_oat_pooled_cond:
            if self.oat_latent_cross_attention_to_expert:
                raise ValueError("use_oat_pooled_cond=True cannot be combined with expert cross-attention.")
            if self.oat_queries_visible_to_action_expert:
                raise ValueError("use_oat_pooled_cond=True requires oat_queries_visible_to_action_expert=False.")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Oat":
        from openpi.models.pi0_oat import Pi0Oat

        return Pi0Oat(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[OATObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = OATObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                oat_latents=(
                    jax.ShapeDtypeStruct([batch_size, self.oat_num_queries, self.oat_latent_dim], jnp.float32)
                    if self.use_oat_latent_alignment
                    else None
                ),
                oat_latent_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.oat_num_queries], jnp.bool_)
                    if self.use_oat_latent_alignment
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
