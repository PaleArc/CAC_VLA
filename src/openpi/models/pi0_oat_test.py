# ruff: noqa: FBT001, FBT003, SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0_oat_config as _pi0_oat_config
from openpi.shared import nnx_utils


def _make_oat_config(**overrides) -> _pi0_oat_config.Pi0OatConfig:
    base_kwargs = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_dim": 8,
        "action_horizon": 4,
        "max_token_len": 16,
        "discrete_state_input": False,
        "use_oat_latent_alignment": True,
        "lambda_latent": 0.1,
        "oat_num_queries": 8,
        "oat_latent_dim": 4,
        "oat_queries_visible_to_action_expert": False,
        "oat_latent_cross_attention_to_expert": True,
        "oat_expert_memory_source_train": "projected_oat",
        "oat_expert_memory_source_infer": "query_hidden",
    }
    base_kwargs.update(overrides)
    return _pi0_oat_config.Pi0OatConfig(**base_kwargs)


def _mask_token_counts(model, timesteps, *, train=False):
    mask = model._build_expert_cross_attn_memory_mask(jnp.asarray(timesteps, dtype=jnp.float32), train=train)
    return jnp.sum(mask, axis=-1)


def test_pi0_oat_raw_space_config_defaults_to_hidden_space():
    config = _make_oat_config()

    assert config.oat_alignment_target_space == "hidden"
    assert config.oat_expert_memory_input_space == "hidden"


def test_pi0_oat_linear_schedule_uses_shifted_token_counts():
    key = jax.random.key(0)
    config = _make_oat_config(oat_expert_memory_time_schedule="ceil_linear")
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.875, 0.75, 0.625, 0.5, 0.375], train=True)

    assert token_counts.tolist() == [4, 5, 6, 7, 8]


def test_pi0_oat_sin_schedule_uses_shifted_token_counts():
    key = jax.random.key(0)
    config = _make_oat_config(oat_expert_memory_time_schedule="ceil_sin")
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.875, 0.5], train=True)

    assert token_counts.tolist() == [5, 8]


def test_pi0_oat_token_shift_can_restore_unshifted_token_counts():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_expert_memory_time_schedule="ceil_linear",
        oat_expert_memory_token_shift=0,
    )
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.875, 0.75, 0.625, 0.5, 0.375], train=True)

    assert token_counts.tolist() == [1, 2, 3, 4, 5]


def test_pi0_oat_resampler_changes_cross_attn_memory_shape():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_expert_memory_query_resampler=True,
        oat_expert_memory_num_queries=4,
        oat_expert_memory_time_schedule="ceil_linear",
    )
    model = config.create(key)

    batch_size = 2
    memory = jnp.ones((batch_size, config.oat_num_queries, 64), dtype=jnp.float32)
    cross_attn_memory = model._build_query_hidden_expert_cross_attn_memory(memory, train=False)
    mask = model._build_expert_cross_attn_memory_mask(jnp.asarray([0.5, 0.25], dtype=jnp.float32), train=False)

    assert cross_attn_memory.shape == (model.paligemma_depth, batch_size, 4, 64)
    assert mask.shape == (batch_size, 4)


def test_pi0_oat_raw_alignment_head_and_memory_projection_shapes():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_alignment_target_space="raw",
        oat_expert_memory_input_space="raw",
    )
    model = config.create(key)

    batch_size = 2
    query_hidden = jnp.ones((batch_size, config.oat_num_queries, model.paligemma_width), dtype=jnp.float32)
    prefix_pre_logits = jnp.ones((batch_size, 5 + config.oat_num_queries, model.paligemma_width), dtype=jnp.float32)
    raw_oat = model._project_query_hidden_to_raw_oat(query_hidden)
    raw_oat_from_prefix = model._extract_query_raw_oat_from_prefix_logits(prefix_pre_logits)
    cross_attn_memory = model._build_query_hidden_expert_cross_attn_memory(raw_oat, train=False)
    raw_oat_for_expert = jnp.ones((batch_size, config.oat_num_queries, config.oat_latent_dim), dtype=jnp.float32)
    raw_cross_attn_memory = model._build_raw_oat_expert_cross_attn_memory(raw_oat_for_expert, train=True)

    assert raw_oat.shape == (batch_size, config.oat_num_queries, config.oat_latent_dim)
    assert raw_oat_from_prefix.shape == (batch_size, config.oat_num_queries, config.oat_latent_dim)
    assert cross_attn_memory.shape == (model.paligemma_depth, batch_size, config.oat_num_queries, model.paligemma_width)
    assert raw_cross_attn_memory.shape == (
        model.paligemma_depth,
        batch_size,
        config.oat_num_queries,
        model.paligemma_width,
    )


def test_pi0_oat_raw_alignment_compute_loss_and_sample_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_alignment_target_space="raw",
        oat_expert_memory_input_space="raw",
        use_oat_alignment_target_stop_gradient=True,
        use_oat_latent_reconstruction=True,
        lambda_oat_recon=0.05,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_noexpert_raw_alignment_compute_loss_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_latent_cross_attention_to_expert=False,
        oat_expert_memory_source_train="query_hidden",
        oat_alignment_target_space="raw",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert loss.shape == (batch_size,)


def test_pi0_oat_log_snr_schedule_is_monotonic():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_expert_memory_query_resampler=True,
        oat_expert_memory_num_queries=4,
        oat_expert_memory_time_schedule="ceil_linear",
        oat_expert_memory_schedule_domain="log_snr",
    )
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.9, 0.7, 0.5, 0.3, 0.1], train=False)

    assert jnp.all(token_counts[1:] >= token_counts[:-1])
    assert token_counts.tolist() == [4, 4, 4, 4, 4]


def test_pi0_oat_staged_schedule_uses_expected_token_counts_in_t_domain():
    key = jax.random.key(0)
    config = _make_oat_config(oat_expert_memory_time_schedule="staged_4_6_8")
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.9, 0.5, 0.1], train=True)

    assert token_counts.tolist() == [4, 6, 8]


def test_pi0_oat_staged_schedule_uses_expected_token_counts_in_log_snr_domain():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_expert_memory_time_schedule="staged_4_6_8",
        oat_expert_memory_schedule_domain="log_snr",
    )
    model = config.create(key)

    token_counts = _mask_token_counts(model, [0.9, 0.5, 0.1], train=False)

    assert token_counts.tolist() == [4, 6, 8]


def test_pi0_oat_schedules_clip_to_available_memory_tokens():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_expert_memory_query_resampler=True,
        oat_expert_memory_num_queries=4,
        oat_expert_memory_time_schedule="staged_4_6_8",
    )
    model = config.create(key)

    staged_counts = _mask_token_counts(model, [0.9, 0.5, 0.1], train=False)
    linear_counts = _mask_token_counts(model, [0.875, 0.75, 0.625], train=False)

    assert staged_counts.tolist() == [4, 4, 4]
    assert linear_counts.tolist() == [4, 4, 4]


def test_pi0_oat_resampler_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_expert_memory_query_resampler=True,
        oat_expert_memory_num_queries=4,
        oat_expert_memory_time_schedule="ceil_linear",
        oat_expert_memory_schedule_domain="log_snr",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_alignment_target_stop_gradient_compute_loss_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(use_oat_alignment_target_stop_gradient=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert config.use_oat_alignment_target_stop_gradient is True
    assert loss.shape == (batch_size,)


def test_pi0_oat_reconstruction_compute_loss_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_alignment_target_stop_gradient=True,
        use_oat_latent_reconstruction=True,
        lambda_oat_recon=0.05,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert config.use_oat_latent_reconstruction is True
    assert config.lambda_oat_recon == pytest.approx(0.05)
    assert loss.shape == (batch_size,)


def test_pi0_oat_poolcond_alignsg_reconstruction_compute_loss_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_latent_cross_attention_to_expert=False,
        use_oat_pooled_cond=True,
        oat_pooled_cond_source_train="projected_oat",
        oat_pooled_cond_source_infer="query_hidden",
        oat_queries_visible_to_action_expert=False,
        use_oat_alignment_target_stop_gradient=True,
        use_oat_latent_reconstruction=True,
        lambda_oat_recon=0.05,
        oat_expert_memory_source_train="query_hidden",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert config.oat_pooled_cond_source_train == "projected_oat"
    assert config.oat_pooled_cond_source_infer == "query_hidden"
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_alignment_target_stop_gradient_blocks_target_side_gradient():
    key = jax.random.key(0)
    batch_size = 2

    def target_grad_norm(use_stop_gradient: bool):
        config = _make_oat_config(use_oat_alignment_target_stop_gradient=use_stop_gradient)
        model = config.create(key)
        hidden_dim = model.latent_query_embeddings.value.shape[-1]
        latent_mask = jnp.ones((batch_size, config.oat_num_queries), dtype=bool)
        prefix_pre_logits = jax.random.normal(
            jax.random.key(1),
            (batch_size, config.oat_num_queries + 2, hidden_dim),
        )
        oat_hidden_targets = jax.random.normal(
            jax.random.key(2),
            (batch_size, config.oat_num_queries, hidden_dim),
        )

        def loss_fn(targets):
            latent_loss, _ = model._compute_latent_alignment_loss(
                targets,
                latent_mask,
                prefix_pre_logits,
                train=False,
            )
            return jnp.mean(latent_loss)

        return jnp.linalg.norm(jax.grad(loss_fn)(oat_hidden_targets))

    assert target_grad_norm(False) > 0.0
    assert jnp.allclose(target_grad_norm(True), 0.0)


def test_pi0_oat_reconstruction_keeps_hidden_side_gradient_with_align_stop_gradient():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_alignment_target_stop_gradient=True,
        use_oat_latent_reconstruction=True,
        lambda_oat_recon=0.05,
    )
    model = config.create(key)
    hidden_dim = model.latent_query_embeddings.value.shape[-1]
    batch_size = 2
    latent_mask = jnp.ones((batch_size, config.oat_num_queries), dtype=bool)
    oat_latents = jax.random.normal(
        jax.random.key(1),
        (batch_size, config.oat_num_queries, config.oat_latent_dim),
    )
    oat_hidden_targets = jax.random.normal(
        jax.random.key(2),
        (batch_size, config.oat_num_queries, hidden_dim),
    )

    def loss_fn(targets):
        recon_loss, _ = model._compute_oat_reconstruction_loss(targets, oat_latents, latent_mask)
        return jnp.mean(recon_loss)

    assert jnp.linalg.norm(jax.grad(loss_fn)(oat_hidden_targets)) > 0.0


def test_pi0_oat_cross_attn_hidden_gate_defaults_off_and_has_no_params():
    key = jax.random.key(0)
    config = _make_oat_config()
    model = config.create(key)

    gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    context_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate.*"))
    kv_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_kv_gate.*"))

    assert config.use_oat_cross_attn_hidden_gate is False
    assert config.use_oat_cross_attn_residual_gate is True
    assert len(gate_state.flat_state()) == 0
    assert config.use_oat_cross_attn_context_gate is False
    assert len(context_gate_state.flat_state()) == 0
    assert config.use_oat_cross_attn_kv_soft_gate is False
    assert len(kv_gate_state.flat_state()) == 0


def test_pi0_oat_cross_attn_hidden_gate_initializes_and_samples():
    key = jax.random.key(0)
    config = _make_oat_config(use_oat_cross_attn_hidden_gate=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert len(gate_state.flat_state()) > 0
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_cross_attn_context_gate_initializes_and_samples():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_cross_attn_hidden_gate=True,
        use_oat_cross_attn_context_gate=True,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    hidden_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    context_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate.*"))
    action_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate_action_proj.*"))
    attn_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate_attn_proj.*"))
    prefix_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate_prefix_proj.*"))
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert len(hidden_gate_state.flat_state()) == 0
    assert len(context_gate_state.flat_state()) > 0
    assert len(action_gate_state.flat_state()) > 0
    assert len(attn_gate_state.flat_state()) > 0
    assert len(prefix_gate_state.flat_state()) > 0
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


@pytest.mark.parametrize(
    ("context_gate_inputs", "expected_proj", "absent_proj"),
    [
        (("action",), "action", ("attn", "prefix")),
        (("attn",), "attn", ("action", "prefix")),
        (("prefix",), "prefix", ("action", "attn")),
        (("action", "attn", "prefix"), None, ()),
    ],
)
def test_pi0_oat_context_gate_inputs_control_initialized_params(
    context_gate_inputs,
    expected_proj,
    absent_proj,
):
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_cross_attn_hidden_gate=True,
        use_oat_cross_attn_context_gate=True,
        oat_cross_attn_context_gate_inputs=context_gate_inputs,
    )
    model = config.create(key)

    score_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_context_gate_score.*"))
    assert len(score_state.flat_state()) > 0

    for proj_name in ("action", "attn", "prefix"):
        proj_state = nnx.state(model).filter(
            nnx_utils.PathRegex(f".*latent_cross_attn_context_gate_{proj_name}_proj.*")
        )
        if expected_proj is None or proj_name == expected_proj:
            assert len(proj_state.flat_state()) > 0
        elif proj_name in absent_proj:
            assert len(proj_state.flat_state()) == 0


def test_pi0_oat_context_gate_inputs_must_be_non_empty_when_enabled():
    with pytest.raises(ValueError, match="must be non-empty"):
        _make_oat_config(
            use_oat_cross_attn_hidden_gate=True,
            use_oat_cross_attn_context_gate=True,
            oat_cross_attn_context_gate_inputs=(),
        )


def test_pi0_oat_context_gate_inputs_rejects_unknown_values():
    with pytest.raises(ValueError, match="must only contain"):
        _make_oat_config(
            use_oat_cross_attn_hidden_gate=True,
            use_oat_cross_attn_context_gate=True,
            oat_cross_attn_context_gate_inputs=("action", "bogus"),
        )


def test_pi0_oat_cross_attn_kv_soft_gate_initializes_and_samples():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_cross_attn_hidden_gate=True,
        use_oat_cross_attn_kv_soft_gate=True,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    hidden_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    kv_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_kv_gate.*"))
    kv_prefix_gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_kv_gate_prefix_proj.*"))
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert len(hidden_gate_state.flat_state()) > 0
    assert len(kv_gate_state.flat_state()) > 0
    assert len(kv_prefix_gate_state.flat_state()) > 0
    assert not config.use_oat_cross_attn_context_gate
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_cross_attn_last_n_layers_initializes_and_samples():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_cross_attn_hidden_gate=True,
        oat_cross_attn_last_n_layers=2,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert len(gate_state.flat_state()) > 0
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_cross_attn_no_residual_gate_ignores_hidden_gate_and_samples():
    key = jax.random.key(0)
    config = _make_oat_config(
        use_oat_cross_attn_residual_gate=False,
        use_oat_cross_attn_hidden_gate=True,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    gate_state = nnx.state(model).filter(nnx_utils.PathRegex(".*latent_cross_attn_hidden_gate.*"))
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert len(gate_state.flat_state()) == 0
    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_oat_scale_aware_noise_is_train_only_and_default_noop():
    key = jax.random.key(0)
    config = _make_oat_config(oat_query_hidden_noise_std=0.5, oat_expert_memory_noise_std=0.5)
    model = config.create(key)
    x = jnp.arange(2 * config.oat_num_queries * 64, dtype=jnp.float32).reshape(2, config.oat_num_queries, 64)

    no_train_noise = model._add_scale_aware_gaussian_noise(x, key, 0.5, train=False)
    no_std_noise = model._add_scale_aware_gaussian_noise(x, key, 0.0, train=True)
    train_noise = model._add_scale_aware_gaussian_noise(x, key, 0.5, train=True)

    assert jnp.allclose(no_train_noise, x)
    assert jnp.allclose(no_std_noise, x)
    assert train_noise.shape == x.shape
    assert train_noise.dtype == x.dtype
    assert not jnp.allclose(train_noise, x)


def test_pi0_oat_expert_memory_noise_probability_is_per_sample():
    key = jax.random.key(0)
    config = _make_oat_config(oat_expert_memory_noise_std=0.5)
    model = config.create(key)
    x = jnp.arange(16 * config.oat_num_queries * 64, dtype=jnp.float32).reshape(16, config.oat_num_queries, 64)

    no_train_noise = model._add_scale_aware_gaussian_noise_with_train_prob(x, key, 0.5, 0.5, train=False)
    no_std_noise = model._add_scale_aware_gaussian_noise_with_train_prob(x, key, 0.0, 0.5, train=True)
    no_prob_noise = model._add_scale_aware_gaussian_noise_with_train_prob(x, key, 0.5, 0.0, train=True)
    full_prob_noise = model._add_scale_aware_gaussian_noise_with_train_prob(x, key, 0.5, 1.0, train=True)

    assert jnp.allclose(no_train_noise, x)
    assert jnp.allclose(no_std_noise, x)
    assert jnp.allclose(no_prob_noise, x)
    assert full_prob_noise.shape == x.shape
    assert full_prob_noise.dtype == x.dtype
    assert jnp.allclose(full_prob_noise, model._add_scale_aware_gaussian_noise(x, key, 0.5, train=True))
    assert not jnp.allclose(full_prob_noise, x)

    mixed_key = None
    expected_mask = None
    for seed in range(100):
        candidate_key = jax.random.key(seed)
        _, gate_rng = jax.random.split(candidate_key)
        mask = jax.random.bernoulli(gate_rng, 0.5, (x.shape[0],))
        if bool(jnp.any(mask)) and bool(jnp.any(~mask)):
            mixed_key = candidate_key
            expected_mask = mask
            break
    assert mixed_key is not None
    assert expected_mask is not None

    partial_noise = model._add_scale_aware_gaussian_noise_with_train_prob(x, mixed_key, 0.5, 0.5, train=True)
    changed_by_sample = jnp.any(partial_noise != x, axis=(1, 2))

    assert partial_noise.shape == x.shape
    assert partial_noise.dtype == x.dtype
    assert bool(jnp.array_equal(changed_by_sample, expected_mask))
    assert bool(jnp.any(changed_by_sample))
    assert bool(jnp.any(~changed_by_sample))


def test_pi0_oat_mixed_expert_memory_compute_loss_and_sample_smoke():
    key = jax.random.key(0)
    config = _make_oat_config(
        oat_expert_memory_source_train="mixed",
        oat_expert_memory_source_infer="query_hidden",
        oat_expert_memory_projected_oat_train_prob=0.8,
        use_oat_alignment_target_stop_gradient=True,
        use_oat_latent_reconstruction=False,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4)

    assert loss.shape == (batch_size,)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert model.oat_expert_memory_source_train == "mixed"
    assert model.oat_expert_memory_source_infer == "query_hidden"
    assert model.oat_expert_memory_projected_oat_train_prob == pytest.approx(0.8)


def test_pi0_oat_config_rejects_negative_noise_std():
    with pytest.raises(ValueError, match="oat_query_hidden_noise_std must be >= 0.0"):
        _make_oat_config(oat_query_hidden_noise_std=-0.01)

    with pytest.raises(ValueError, match="oat_expert_memory_noise_std must be >= 0.0"):
        _make_oat_config(oat_expert_memory_noise_std=-0.01)


def test_pi0_oat_config_validates_expert_memory_noise_probability():
    with pytest.raises(ValueError, match="oat_expert_memory_noise_train_prob must be in"):
        _make_oat_config(oat_expert_memory_noise_train_prob=-0.01)

    with pytest.raises(ValueError, match="oat_expert_memory_noise_train_prob must be in"):
        _make_oat_config(oat_expert_memory_noise_train_prob=1.01)


def test_pi0_oat_config_validates_raw_space_options():
    with pytest.raises(ValueError, match="oat_alignment_target_space must be one of"):
        _make_oat_config(oat_alignment_target_space="bad")

    with pytest.raises(ValueError, match="oat_expert_memory_input_space must be one of"):
        _make_oat_config(oat_expert_memory_input_space="bad")

    with pytest.raises(ValueError, match="raw.*mixed expert memory"):
        _make_oat_config(
            oat_expert_memory_input_space="raw",
            oat_expert_memory_source_train="mixed",
        )


def test_pi0_oat_config_validates_mixed_projected_oat_probability():
    with pytest.raises(ValueError, match="oat_expert_memory_projected_oat_train_prob must be in"):
        _make_oat_config(
            oat_expert_memory_source_train="mixed",
            oat_expert_memory_projected_oat_train_prob=-0.01,
        )

    with pytest.raises(ValueError, match="oat_expert_memory_projected_oat_train_prob must be in"):
        _make_oat_config(
            oat_expert_memory_source_train="mixed",
            oat_expert_memory_projected_oat_train_prob=1.01,
        )

    query_config = _make_oat_config(
        oat_expert_memory_source_train="query_hidden",
        oat_expert_memory_projected_oat_train_prob=0.25,
    )
    assert query_config.oat_expert_memory_source_train == "query_hidden"
    assert query_config.oat_expert_memory_projected_oat_train_prob == pytest.approx(0.25)


def test_pi0_oat_config_rejects_invalid_expert_memory_query_count():
    with pytest.raises(ValueError, match="oat_expert_memory_num_queries must be >= 1"):
        _make_oat_config(use_oat_expert_memory_query_resampler=True, oat_expert_memory_num_queries=0)


def test_pi0_oat_config_rejects_negative_expert_memory_token_shift():
    with pytest.raises(ValueError, match="oat_expert_memory_token_shift must be >= 0"):
        _make_oat_config(oat_expert_memory_token_shift=-1)


def test_pi0_oat_config_rejects_invalid_cross_attn_last_n_layers():
    with pytest.raises(ValueError, match="oat_cross_attn_last_n_layers must be a positive integer or None"):
        _make_oat_config(oat_cross_attn_last_n_layers=0)

    config = _make_oat_config(oat_cross_attn_last_n_layers=5)
    with pytest.raises(ValueError, match=r"oat_cross_attn_last_n_layers must be in \[1, 4\]"):
        config.create(jax.random.key(0))
