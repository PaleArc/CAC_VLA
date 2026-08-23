from __future__ import annotations

import argparse
from collections.abc import Iterator
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.folder_dataset import write_metadata
import torch
from tqdm import tqdm

DEFAULT_DATASETS = [
    "libero_mix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment LIBERO-plus RLDS datasets with OAT latent targets.")
    parser.add_argument(
        "--source-rlds-root",
        type=Path,
        required=True,
        help="Root directory containing the source LIBERO-plus RLDS datasets.",
    )
    parser.add_argument(
        "--source-libero-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing raw LIBERO-plus HDF5 files. If provided, episode metadata file paths "
            "are rewritten to matching local HDF5 paths by basename."
        ),
    )
    parser.add_argument(
        "--output-rlds-root",
        type=Path,
        required=True,
        help="Root directory where the OAT-augmented LIBERO-plus RLDS datasets will be written.",
    )
    parser.add_argument(
        "--oat-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "external" / "oat",
        help="Path to the OAT repository root.",
    )
    parser.add_argument(
        "--oat-checkpoint",
        type=Path,
        required=True,
        help="Path to a trained OAT tokenizer checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Future action horizon used for OAT latent extraction.",
    )
    parser.add_argument(
        "--action-stride",
        type=int,
        default=None,
        help="Stride between future actions when constructing OAT action chunks. "
        "If omitted, try to infer it from the tokenizer checkpoint config and fall back to 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for OAT latent extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for OAT inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    return parser.parse_args()


def _load_oat_tokenizer(oat_root: Path, checkpoint: Path, device: str):
    try:
        import dill
        import hydra
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing OAT tokenizer dependency. Install dill and hydra-core.") from exc

    if str(oat_root) not in sys.path:
        sys.path.insert(0, str(oat_root))
    with checkpoint.open("rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    state_key = (
        "ema_model"
        if bool(getattr(cfg.training, "use_ema", False)) and "ema_model" in payload["state_dicts"]
        else "model"
    )
    tokenizer.load_state_dict(payload["state_dicts"][state_key])
    tokenizer.eval()
    tokenizer.to(device)
    return tokenizer, cfg


def _build_file_index(libero_root: Path | None) -> dict[str, str]:
    if libero_root is None or not libero_root.exists():
        return {}
    index: dict[str, str] = {}
    for path in libero_root.rglob("*.hdf5"):
        index[path.name] = str(path)
    return index


def _build_augmented_features(source_builder, num_queries: int, latent_dim: int):
    source_features = source_builder.info.features
    source_step_features = dict(source_features["steps"].feature.items())
    source_step_features["oat_latents"] = tfds.features.Tensor(shape=(num_queries, latent_dim), dtype=np.float32)
    source_step_features["oat_latent_mask"] = tfds.features.Tensor(shape=(num_queries,), dtype=np.bool_)
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(tfds.features.FeaturesDict(source_step_features)),
            "episode_metadata": source_features["episode_metadata"],
        }
    )


def _build_action_chunks(actions: np.ndarray, horizon: int, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError(f"action_stride must be >= 1, got {stride}")
    traj_len = actions.shape[0]
    gather_idx = np.arange(traj_len)[:, None] + np.arange(horizon)[None, :] * stride
    gather_idx = np.clip(gather_idx, 0, traj_len - 1)
    return actions[gather_idx]


def _compute_oat_latents(
    tokenizer,
    actions: np.ndarray,
    horizon: int,
    stride: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    action_chunks = _build_action_chunks(actions.astype(np.float32), horizon, stride)
    latent_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, action_chunks.shape[0], batch_size):
            batch = torch.from_numpy(action_chunks[start : start + batch_size]).to(device)
            latents, _ = tokenizer.encode(batch)
            latent_batches.append(latents.detach().cpu().numpy().astype(np.float32))
    latents = np.concatenate(latent_batches, axis=0)
    latent_mask = np.ones((latents.shape[0], latents.shape[1]), dtype=np.bool_)
    return latents, latent_mask


def _iter_serialized_examples(shard_paths: list[Path], max_episodes: int | None = None) -> Iterator[bytes]:
    seen = 0
    for shard_path in shard_paths:
        dataset = tf.data.TFRecordDataset([str(shard_path)])
        for raw_example in dataset:
            if max_episodes is not None and seen >= max_episodes:
                return
            yield bytes(raw_example.numpy())
            seen += 1


def _extract_actions(example: tf.train.Example) -> np.ndarray:
    features = example.features.feature
    traj_len = len(features["steps/is_first"].int64_list.value)
    flat_actions = np.asarray(features["steps/action"].float_list.value, dtype=np.float32)
    if traj_len == 0:
        raise ValueError("Encountered empty trajectory while extracting actions.")
    if flat_actions.size % traj_len != 0:
        raise ValueError(f"Action feature length {flat_actions.size} is not divisible by trajectory length {traj_len}.")
    action_dim = flat_actions.size // traj_len
    return flat_actions.reshape(traj_len, action_dim)


def _rewrite_file_path(example: tf.train.Example, file_index: dict[str, str]) -> None:
    feature = example.features.feature["episode_metadata/file_path"]
    if not feature.bytes_list.value:
        return
    file_path = feature.bytes_list.value[0].decode()
    basename = os.path.basename(file_path)
    rewritten = file_index.get(basename, file_path)
    feature.bytes_list.value[:] = [rewritten.encode()]


def _augment_serialized_example(
    serialized_example: bytes,
    tokenizer,
    *,
    action_horizon: int,
    action_stride: int,
    batch_size: int,
    device: str,
    file_index: dict[str, str],
) -> tuple[bytes, int, int]:
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    actions = _extract_actions(example)
    oat_latents, oat_latent_mask = _compute_oat_latents(
        tokenizer,
        actions=actions,
        horizon=action_horizon,
        stride=action_stride,
        batch_size=batch_size,
        device=device,
    )

    example.features.feature["steps/oat_latents"].float_list.value[:] = oat_latents.reshape(-1).tolist()
    example.features.feature["steps/oat_latent_mask"].int64_list.value[:] = (
        oat_latent_mask.astype(np.int64).reshape(-1).tolist()
    )
    _rewrite_file_path(example, file_index)

    num_queries, latent_dim = oat_latents.shape[1], oat_latents.shape[2]
    return example.SerializeToString(), num_queries, latent_dim


def _get_source_shard_paths(source_version_dir: Path, source_builder) -> list[Path]:
    shard_paths = sorted(source_version_dir.glob(f"{source_builder.info.name}-train.tfrecord-*"))
    if not shard_paths:
        raise FileNotFoundError(f"No TFRecord shards found under {source_version_dir}")
    return shard_paths


def _iter_episodes(builder, max_episodes: int | None = None) -> Iterator[dict]:
    dataset = builder.as_dataset(split="train")
    for idx, episode in enumerate(tfds.as_numpy(dataset)):
        if max_episodes is not None and idx >= max_episodes:
            break
        yield episode


def _write_augmented_dataset(
    source_builder,
    source_version_dir: Path,
    output_version_dir: Path,
    tokenizer,
    *,
    action_horizon: int,
    action_stride: int,
    batch_size: int,
    device: str,
    file_index: dict[str, str],
    max_episodes: int | None,
) -> tuple[int, int]:
    shard_paths = _get_source_shard_paths(source_version_dir, source_builder)
    serialized_examples = _iter_serialized_examples(shard_paths, max_episodes=max_episodes)

    split_info = source_builder.info.splits["train"]
    source_shard_lengths = list(split_info.shard_lengths)
    shard_lengths = [max_episodes] if max_episodes is not None else source_shard_lengths

    num_shards = len(shard_lengths)
    shard_template = f"{source_builder.info.name}-train.tfrecord-{{:05d}}-of-{num_shards:05d}"

    total_written = 0
    num_queries = latent_dim = None
    features = None
    writers: list[tf.io.TFRecordWriter] = []
    try:
        shard_idx = 0
        current_writer = tf.io.TFRecordWriter(str(output_version_dir / shard_template.format(shard_idx)))
        writers.append(current_writer)
        remaining_in_shard = shard_lengths[shard_idx]

        for serialized_example in tqdm(
            serialized_examples, desc=f"Converting {output_version_dir.parent.name}", unit="episode"
        ):
            augmented_serialized, current_num_queries, current_latent_dim = _augment_serialized_example(
                serialized_example,
                tokenizer,
                action_horizon=action_horizon,
                action_stride=action_stride,
                batch_size=batch_size,
                device=device,
                file_index=file_index,
            )
            if num_queries is None or latent_dim is None:
                num_queries, latent_dim = current_num_queries, current_latent_dim
                features = _build_augmented_features(source_builder, num_queries=num_queries, latent_dim=latent_dim)
            elif (num_queries, latent_dim) != (current_num_queries, current_latent_dim):
                raise RuntimeError(
                    "Encountered inconsistent OAT latent shape while augmenting dataset: "
                    f"expected {(num_queries, latent_dim)}, got {(current_num_queries, current_latent_dim)}."
                )

            if remaining_in_shard == 0:
                current_writer.close()
                shard_idx += 1
                current_writer = tf.io.TFRecordWriter(str(output_version_dir / shard_template.format(shard_idx)))
                writers.append(current_writer)
                remaining_in_shard = shard_lengths[shard_idx]

            current_writer.write(augmented_serialized)
            total_written += 1
            remaining_in_shard -= 1
    finally:
        for writer in writers:
            writer.close()

    if num_queries is None or latent_dim is None:
        raise RuntimeError(f"No episodes were written to {output_version_dir}")

    write_metadata(
        data_dir=output_version_dir,
        features=features,
        filename_template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        split_infos=None,
        check_data=True,
        description=source_builder.info.description,
        homepage=source_builder.info.homepage,
        citation=source_builder.info.citation,
    )

    return num_queries, latent_dim


def _copy_norm_stats(source_version_dir: Path, output_version_dir: Path) -> None:
    src = source_version_dir / "norm_stats.json"
    if src.exists():
        shutil.copy2(src, output_version_dir / "norm_stats.json")


def main() -> None:
    args = parse_args()

    args.output_rlds_root.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_cfg = _load_oat_tokenizer(args.oat_root, args.oat_checkpoint, args.device)
    inferred_action_stride = (
        getattr(getattr(getattr(tokenizer_cfg, "task", None), "tokenizer", None), "dataset", None).action_stride
        if getattr(getattr(getattr(tokenizer_cfg, "task", None), "tokenizer", None), "dataset", None) is not None
        and hasattr(
            getattr(getattr(getattr(tokenizer_cfg, "task", None), "tokenizer", None), "dataset", None), "action_stride"
        )
        else 1
    )
    action_stride = args.action_stride if args.action_stride is not None else inferred_action_stride
    file_index = _build_file_index(args.source_libero_root)

    for dataset_name in args.datasets:
        source_version_dir = args.source_rlds_root / dataset_name / "1.0.0"
        if not source_version_dir.exists():
            raise FileNotFoundError(f"Source RLDS directory not found: {source_version_dir}")

        output_version_dir = args.output_rlds_root / dataset_name / "1.0.0"
        if output_version_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory already exists: {output_version_dir}")
            shutil.rmtree(output_version_dir.parent)
        output_version_dir.mkdir(parents=True, exist_ok=True)

        source_builder = tfds.builder_from_directory(str(source_version_dir))
        num_queries, latent_dim = _write_augmented_dataset(
            source_builder,
            source_version_dir,
            output_version_dir,
            tokenizer,
            action_horizon=args.action_horizon,
            action_stride=action_stride,
            batch_size=args.batch_size,
            device=args.device,
            file_index=file_index,
            max_episodes=args.max_episodes,
        )
        _copy_norm_stats(source_version_dir, output_version_dir)

        print(
            f"[done] {dataset_name}: wrote OAT-augmented RLDS to {output_version_dir} "
            f"(oat_num_queries={num_queries}, oat_latent_dim={latent_dim}, action_stride={action_stride})"
        )


if __name__ == "__main__":
    main()
