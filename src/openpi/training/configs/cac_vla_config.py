import os
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_oat_config as pi0_oat_config
import openpi.training.config as _config
from openpi.training.configs import rlds
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

_PI05_BASE_PARAMS = os.environ.get(
    "OPENPI_PI05_BASE_PARAMS",
    "gs://openpi-assets/checkpoints/pi05_base/params",
)
_OAT_CROSS_ATTN_MISSING_REGEX = (
    ".*(lora.*|latent_query_embeddings.*|oat_to_hidden_proj/.*|query_memory_norm/.*|"
    "query_memory_proj/.*|pre_latent_cross_attention_norm_1/.*|latent_cross_attn_1/.*|"
    "latent_cross_attn_context_gate_action_proj_1/.*|latent_cross_attn_context_gate_attn_proj_1/.*|"
    "latent_cross_attn_context_gate_prefix_proj_1/.*|latent_cross_attn_context_gate_score_1/.*|"
    "query_oat_norm/.*|query_oat_proj/.*)"
)
_OAT_ALIGNMENT_ONLY_MISSING_REGEX = (
    ".*(lora.*|latent_query_embeddings.*|oat_to_hidden_proj/.*|query_memory_norm/.*|"
    "query_memory_proj/.*|query_oat_norm/.*|query_oat_proj/.*)"
)


def _data_dir(env_name: str, relative_default: str) -> str:
    return os.environ.get(env_name, str(pathlib.Path("data") / "rlds" / relative_default))


def _pi05_config(*, discrete_state_input: bool = False) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=discrete_state_input,
    )


def _oat_config(
    *,
    discrete_state_input: bool = False,
    cross_attention: bool = True,
    direct_residual: bool = False,
) -> pi0_oat_config.Pi0OatConfig:
    return pi0_oat_config.Pi0OatConfig(
        pi05=True,
        action_horizon=10,
        discrete_state_input=discrete_state_input,
        use_oat_latent_alignment=True,
        lambda_latent=0.1,
        oat_num_queries=8,
        oat_latent_dim=4,
        oat_queries_visible_to_action_expert=False,
        oat_latent_cross_attention_to_expert=cross_attention,
        oat_expert_memory_source_train="projected_oat" if cross_attention else "query_hidden",
        oat_expert_memory_source_infer="query_hidden",
        oat_expert_memory_dropout_rate=0.1,
        use_oat_cross_attn_residual_gate=not direct_residual,
        use_oat_cross_attn_hidden_gate=cross_attention and not direct_residual,
        use_oat_cross_attn_context_gate=cross_attention and not direct_residual,
        oat_cross_attn_context_gate_inputs=("action",)
        if cross_attention and not direct_residual
        else ("action", "attn", "prefix"),
        oat_alignment_target_space="raw",
        oat_expert_memory_input_space="raw",
        oat_query_hidden_noise_std=0.01 if cross_attention else 0.0,
        oat_expert_memory_noise_std=0.05 if cross_attention else 0.0,
    )


def _train_config(
    *,
    name: str,
    model,
    data: _config.DataConfigFactory,
    num_train_steps: int = 30_001,
    peak_lr: float = 1.25e-5,
    missing_regex: str = ".*lora.*",
) -> _config.TrainConfig:
    return _config.TrainConfig(
        name=name,
        project_name="cac_vla",
        model=model,
        data=data,
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=peak_lr,
            decay_steps=1_000_000,
            decay_lr=peak_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _PI05_BASE_PARAMS,
            missing_regex=missing_regex,
        ),
        num_train_steps=num_train_steps,
        save_interval=10_000,
    )


def _libero_data(root: str) -> rlds.RLDSLiberoDataConfig:
    return rlds.RLDSLiberoDataConfig(
        repo_id="libero",
        assets=_config.AssetsConfig(asset_id="libero"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def _libero_oat_data(root: str) -> rlds.RLDSLiberoOATDataConfig:
    return rlds.RLDSLiberoOATDataConfig(
        repo_id="libero",
        assets=_config.AssetsConfig(asset_id="libero"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def _libero_plus_data(root: str) -> rlds.RLDSLiberoPlusDataConfig:
    return rlds.RLDSLiberoPlusDataConfig(
        repo_id="libero",
        assets=_config.AssetsConfig(asset_id="libero"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def _libero_plus_oat_data(root: str) -> rlds.RLDSLiberoPlusOATDataConfig:
    return rlds.RLDSLiberoPlusOATDataConfig(
        repo_id="libero",
        assets=_config.AssetsConfig(asset_id="libero"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def _calvin_data(root: str) -> rlds.RLDSCalvinDataConfig:
    return rlds.RLDSCalvinDataConfig(
        repo_id="calvin",
        assets=_config.AssetsConfig(asset_id="calvin"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def _calvin_oat_data(root: str) -> rlds.RLDSCalvinOATDataConfig:
    return rlds.RLDSCalvinOATDataConfig(
        repo_id="calvin",
        assets=_config.AssetsConfig(asset_id="calvin"),
        rlds_data_dir=root,
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def get_configs() -> list[_config.TrainConfig]:
    libero_root = _data_dir("OPENPI_LIBERO_RLDS_DIR", "libero_oat_h10")
    libero_oat_root = _data_dir("OPENPI_LIBERO_OAT_RLDS_DIR", "libero_oat_h10")
    libero_plus_root = _data_dir("OPENPI_LIBERO_PLUS_RLDS_DIR", "libero_plus_oat_h10")
    libero_plus_h10 = _data_dir("OPENPI_LIBERO_PLUS_OAT_H10_DIR", "libero_plus_oat_h10")
    libero_plus_h20 = _data_dir("OPENPI_LIBERO_PLUS_OAT_H20_DIR", "libero_plus_oat_h20")
    libero_plus_h30 = _data_dir("OPENPI_LIBERO_PLUS_OAT_H30_DIR", "libero_plus_oat_h30")
    calvin_root = _data_dir("OPENPI_CALVIN_RLDS_DIR", "calvin_oat")
    calvin_oat_root = _data_dir("OPENPI_CALVIN_OAT_RLDS_DIR", "calvin_oat")

    return [
        _train_config(
            name="pi05_libero_plus_oat_rawalign_only_action_h10",
            model=_oat_config(),
            data=_libero_plus_oat_data(libero_plus_h10),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_libero_plus_oat_rawalign_only_action_h20",
            model=_oat_config(),
            data=_libero_plus_oat_data(libero_plus_h20),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_libero_plus_oat_rawalign_only_action_h30",
            model=_oat_config(),
            data=_libero_plus_oat_data(libero_plus_h30),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_base_libero_plus_rlds",
            model=_pi05_config(),
            data=_libero_plus_data(libero_plus_root),
        ),
        _train_config(
            name="pi05_libero_rlds",
            model=_pi05_config(),
            data=_libero_data(libero_root),
            num_train_steps=60_001,
        ),
        _train_config(
            name="pi05_libero_oat_rawalign_only_action",
            model=_oat_config(),
            data=_libero_oat_data(libero_oat_root),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_libero_plus_oat_rawalign_noexpert_h10",
            model=_oat_config(cross_attention=False),
            data=_libero_plus_oat_data(libero_plus_h10),
            peak_lr=1e-5,
            missing_regex=_OAT_ALIGNMENT_ONLY_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_libero_plus_oat_rawalign_directresidual_h10",
            model=_oat_config(direct_residual=True),
            data=_libero_plus_oat_data(libero_plus_h10),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_calvin_oat_rawalign_only_action",
            model=_oat_config(discrete_state_input=True),
            data=_calvin_oat_data(calvin_oat_root),
            missing_regex=_OAT_CROSS_ATTN_MISSING_REGEX,
        ),
        _train_config(
            name="pi05_calvin_rlds",
            model=_pi05_config(discrete_state_input=True),
            data=_calvin_data(calvin_root),
        ),
    ]
