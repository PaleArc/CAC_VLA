from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
from typing_extensions import override

import openpi.models.model as _model
import openpi.policies.libero_oat_policy as libero_oat_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.transforms as _transforms

LIBERO_DATASETS = (
    droid_rlds_dataset.RLDSDataset(name="libero_10_no_noops", version="1.0.0", weight=0.25),
    droid_rlds_dataset.RLDSDataset(name="libero_goal_no_noops", version="1.0.0", weight=0.25),
    droid_rlds_dataset.RLDSDataset(name="libero_object_no_noops", version="1.0.0", weight=0.25),
    droid_rlds_dataset.RLDSDataset(name="libero_spatial_no_noops", version="1.0.0", weight=0.25),
)
LIBERO_PLUS_DATASETS = (droid_rlds_dataset.RLDSDataset(name="libero_mix", version="1.0.0", weight=1.0),)
CALVIN_DATASETS = (droid_rlds_dataset.RLDSDataset(name="calvin", version="1.0.0", weight=1.0),)


def _load_combined_norm_stats(
    rlds_data_dir: pathlib.Path,
    datasets: Sequence[droid_rlds_dataset.RLDSDataset],
) -> dict[str, _transforms.NormStats]:
    stats_acc: dict[str, dict[str, Any]] = {}

    for dataset in datasets:
        stats_path = rlds_data_dir / dataset.name / dataset.version / "norm_stats.json"
        payload = json.loads(stats_path.read_text())["norm_stats"]
        dataset_num_transitions = payload.get("actions", {}).get("num_transitions")
        for key, values in payload.items():
            num_transitions = values.get("num_transitions", dataset_num_transitions)
            if num_transitions is None:
                if len(datasets) != 1:
                    raise ValueError(f"{stats_path} must define num_transitions when combining multiple RLDS datasets.")
                weight = 1.0
            else:
                weight = float(num_transitions)

            mean = np.asarray(values["mean"], dtype=np.float32)
            std = np.asarray(values["std"], dtype=np.float32)
            q01 = np.asarray(values["q01"], dtype=np.float32) if values.get("q01") is not None else None
            q99 = np.asarray(values["q99"], dtype=np.float32) if values.get("q99") is not None else None
            second_moment = std**2 + mean**2
            bucket = stats_acc.setdefault(
                key,
                {
                    "weight": 0.0,
                    "mean": np.zeros_like(mean),
                    "second_moment": np.zeros_like(second_moment),
                    "q01": None if q01 is None else np.zeros_like(q01),
                    "q99": None if q99 is None else np.zeros_like(q99),
                },
            )
            bucket["weight"] += weight
            bucket["mean"] += weight * mean
            bucket["second_moment"] += weight * second_moment
            if q01 is not None and bucket["q01"] is not None:
                bucket["q01"] += weight * q01
            if q99 is not None and bucket["q99"] is not None:
                bucket["q99"] += weight * q99

    combined = {}
    for key, values in stats_acc.items():
        weight = values["weight"]
        mean = values["mean"] / weight
        second_moment = values["second_moment"] / weight
        combined[key] = _normalize.NormStats(
            mean=mean,
            std=np.sqrt(np.maximum(0.0, second_moment - mean**2)),
            q01=None if values["q01"] is None else values["q01"] / weight,
            q99=None if values["q99"] is None else values["q99"] / weight,
        )
    return combined


def _create_data_config(
    factory: "_RLDSRobotDataConfig",
    assets_dirs: pathlib.Path,
    model_config: _model.BaseModelConfig,
    *,
    oat: bool,
    load_norm_stats: bool,
) -> _config.DataConfig:
    if factory.rlds_data_dir is None:
        raise ValueError("rlds_data_dir must be set for the RLDS data loader.")

    repack_mapping = {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    }
    if oat:
        repack_mapping.update(
            {
                "oat_latents": "oat_latents",
                "oat_latent_mask": "oat_latent_mask",
            }
        )
    repack_transforms = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])

    if oat:
        inputs = libero_oat_policy.LiberoOATInputs(model_type=model_config.model_type)
        outputs = libero_oat_policy.LiberoOATOutputs()
    else:
        inputs = libero_policy.LiberoInputs(model_type=model_config.model_type)
        outputs = libero_policy.LiberoOutputs()
    data_transforms = _transforms.Group(inputs=[inputs], outputs=[outputs])

    if factory.extra_delta_transform:
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

    norm_stats = (
        _load_combined_norm_stats(pathlib.Path(factory.rlds_data_dir), factory.datasets) if load_norm_stats else None
    )
    return dataclasses.replace(
        factory.create_base_config(assets_dirs, model_config),
        norm_stats=norm_stats,
        repack_transforms=repack_transforms,
        data_transforms=data_transforms,
        model_transforms=_config.ModelTransformFactory()(model_config),
        rlds_data_dir=factory.rlds_data_dir,
        rlds_dataset_type="libero_oat" if oat else "libero",
        observation_type="oat" if oat else "default",
        datasets=factory.datasets,
    )


@dataclasses.dataclass(frozen=True)
class _RLDSRobotDataConfig(_config.DataConfigFactory):
    rlds_data_dir: str | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = LIBERO_DATASETS
    extra_delta_transform: bool = False
    oat: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return _create_data_config(self, assets_dirs, model_config, oat=self.oat, load_norm_stats=True)

    def create_for_norm_stats(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> _config.DataConfig:
        return _create_data_config(self, assets_dirs, model_config, oat=self.oat, load_norm_stats=False)


@dataclasses.dataclass(frozen=True)
class RLDSLiberoDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = LIBERO_DATASETS


@dataclasses.dataclass(frozen=True)
class RLDSLiberoPlusDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = LIBERO_PLUS_DATASETS


@dataclasses.dataclass(frozen=True)
class RLDSCalvinDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = CALVIN_DATASETS


@dataclasses.dataclass(frozen=True)
class RLDSLiberoOATDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = LIBERO_DATASETS
    oat: bool = True


@dataclasses.dataclass(frozen=True)
class RLDSLiberoPlusOATDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = LIBERO_PLUS_DATASETS
    oat: bool = True


@dataclasses.dataclass(frozen=True)
class RLDSCalvinOATDataConfig(_RLDSRobotDataConfig):
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = CALVIN_DATASETS
    oat: bool = True
