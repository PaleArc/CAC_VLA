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
OATMode: TypeAlias = Literal["only_action", "direct_residual", "noexpert"]


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
    oat_mode: OATMode = "only_action"
    lambda_latent: float = 0.0
    oat_num_queries: int | None = None
    oat_latent_dim: int | None = None
    use_oat_alignment_target_stop_gradient: bool = False
    use_oat_latent_reconstruction: bool = False
    lambda_oat_recon: float = 0.0
    oat_alignment_target_space: Literal["raw"] = "raw"
    oat_expert_memory_input_space: Literal["raw"] = "raw"
    oat_query_hidden_noise_std: float = 0.0
    oat_expert_memory_noise_std: float = 0.0
    oat_expert_memory_noise_train_prob: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0OatConfig currently only supports pi05=True.")
        if self.oat_mode not in {"only_action", "direct_residual", "noexpert"}:
            raise ValueError(f"oat_mode must be one of only_action, direct_residual, noexpert, got {self.oat_mode!r}.")
        if self.oat_mode != "noexpert" and not self.use_oat_latent_alignment:
            raise ValueError(f"oat_mode={self.oat_mode!r} requires use_oat_latent_alignment=True.")
        if self.use_oat_latent_alignment and (self.oat_num_queries is None or self.oat_latent_dim is None):
            raise ValueError("oat_num_queries and oat_latent_dim are required when use_oat_latent_alignment=True.")
        if self.oat_num_queries is not None and self.oat_num_queries < 1:
            raise ValueError("oat_num_queries must be >= 1.")
        if self.oat_latent_dim is not None and self.oat_latent_dim < 1:
            raise ValueError("oat_latent_dim must be >= 1.")
        if self.lambda_latent < 0.0 or self.lambda_oat_recon < 0.0:
            raise ValueError("latent loss weights must be >= 0.0.")
        if self.use_oat_latent_reconstruction and not self.use_oat_latent_alignment:
            raise ValueError("use_oat_latent_reconstruction=True requires use_oat_latent_alignment=True.")
        if self.oat_alignment_target_space != "raw" or self.oat_expert_memory_input_space != "raw":
            raise ValueError("OAT uses raw latent alignment and raw expert memory only.")
        if self.oat_query_hidden_noise_std < 0.0 or self.oat_expert_memory_noise_std < 0.0:
            raise ValueError("OAT noise standard deviations must be >= 0.0.")
        if not 0.0 <= self.oat_expert_memory_noise_train_prob <= 1.0:
            raise ValueError("oat_expert_memory_noise_train_prob must be in [0.0, 1.0].")
        if not self.use_oat_latent_alignment:
            object.__setattr__(self, "lambda_latent", 0.0)
            object.__setattr__(self, "use_oat_latent_reconstruction", False)
            object.__setattr__(self, "lambda_oat_recon", 0.0)
            object.__setattr__(self, "oat_query_hidden_noise_std", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_std", 0.0)
            object.__setattr__(self, "oat_expert_memory_noise_train_prob", 0.0)

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
