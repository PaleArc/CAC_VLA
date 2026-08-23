import pathlib

import pytest

import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_oat_config as pi0_oat_config
import openpi.training.config as _config
from openpi.training.configs import palearc

EXPECTED_CONFIG_NAMES = (
    "pi05_libero_plus_oat_rawalign_only_action_h10",
    "pi05_libero_plus_oat_rawalign_only_action_h20",
    "pi05_libero_plus_oat_rawalign_only_action_h30",
    "pi05_base_libero_plus_rlds",
    "pi05_libero_rlds",
    "pi05_libero_oat_rawalign_only_action",
    "pi05_libero_plus_oat_rawalign_noexpert_h10",
    "pi05_libero_plus_oat_rawalign_directresidual_h10",
    "pi05_calvin_oat_rawalign_only_action",
    "pi05_calvin_rlds",
)


def test_project_config_registry_is_intentionally_small():
    configs = palearc.get_configs()

    assert tuple(config.name for config in configs) == EXPECTED_CONFIG_NAMES
    assert all(_config.get_config(name).name == name for name in EXPECTED_CONFIG_NAMES)


@pytest.mark.parametrize(
    ("name", "data_suffix"),
    [
        ("pi05_libero_plus_oat_rawalign_only_action_h10", "libero_plus_oat_h10"),
        ("pi05_libero_plus_oat_rawalign_only_action_h20", "libero_plus_oat_h20"),
        ("pi05_libero_plus_oat_rawalign_only_action_h30", "libero_plus_oat_h30"),
        ("pi05_libero_oat_rawalign_only_action", "libero_oat_h10"),
        ("pi05_calvin_oat_rawalign_only_action", "calvin_oat"),
    ],
)
def test_raw_alignment_configs_preserve_model_behavior(name: str, data_suffix: str):
    config = _config.get_config(name)

    assert isinstance(config.model, pi0_oat_config.Pi0OatConfig)
    assert config.model.use_oat_latent_alignment is True
    assert config.model.oat_latent_cross_attention_to_expert is True
    assert config.model.oat_cross_attn_context_gate_inputs == ("action",)
    assert config.model.oat_alignment_target_space == "raw"
    assert config.model.oat_expert_memory_input_space == "raw"
    assert config.model.oat_query_hidden_noise_std == pytest.approx(0.01)
    assert config.model.oat_expert_memory_noise_std == pytest.approx(0.05)
    assert pathlib.Path(config.data.rlds_data_dir).name == data_suffix
    assert not pathlib.Path(config.data.rlds_data_dir).is_absolute()


def test_noexpert_config_disables_cross_attention():
    config = _config.get_config("pi05_libero_plus_oat_rawalign_noexpert_h10")

    assert isinstance(config.model, pi0_oat_config.Pi0OatConfig)
    assert config.model.oat_latent_cross_attention_to_expert is False
    assert config.model.use_oat_cross_attn_hidden_gate is False
    assert config.model.use_oat_cross_attn_context_gate is False
    assert config.model.oat_alignment_target_space == "raw"
    assert config.lr_schedule.peak_lr == pytest.approx(1e-5)


def test_direct_residual_config_disables_gates():
    config = _config.get_config("pi05_libero_plus_oat_rawalign_directresidual_h10")

    assert isinstance(config.model, pi0_oat_config.Pi0OatConfig)
    assert config.model.oat_latent_cross_attention_to_expert is True
    assert config.model.use_oat_cross_attn_residual_gate is False
    assert config.model.use_oat_cross_attn_hidden_gate is False
    assert config.model.use_oat_cross_attn_context_gate is False


@pytest.mark.parametrize(
    ("name", "discrete_state_input"),
    [
        ("pi05_base_libero_plus_rlds", False),
        ("pi05_libero_rlds", False),
        ("pi05_calvin_rlds", True),
    ],
)
def test_baseline_configs_use_pi05(name: str, discrete_state_input):
    config = _config.get_config(name)

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05 is True
    assert config.model.discrete_state_input is discrete_state_input


def test_calvin_oat_config_uses_discrete_state_input():
    config = _config.get_config("pi05_calvin_oat_rawalign_only_action")

    assert config.model.discrete_state_input is True
