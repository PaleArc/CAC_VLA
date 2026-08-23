from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import io
import json
import logging
from pathlib import Path
import random
import struct

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
import numpy as np
from PIL import Image

from openpi.training.droid_rlds_dataset import RLDSDataset


def _build_example_message_class():
    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "openpi_tf_example.proto"
    file_desc.package = "tensorflow"
    file_desc.syntax = "proto3"

    for name, field_type in (
        ("BytesList", descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
        ("FloatList", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        ("Int64List", descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
    ):
        message = file_desc.message_type.add()
        message.name = name
        field = message.field.add()
        field.name = "value"
        field.number = 1
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        field.type = field_type
        if field_type != descriptor_pb2.FieldDescriptorProto.TYPE_BYTES:
            field.options.packed = True

    feature = file_desc.message_type.add()
    feature.name = "Feature"
    oneof = feature.oneof_decl.add()
    oneof.name = "kind"
    for idx, (name, type_name) in enumerate(
        (("bytes_list", "BytesList"), ("float_list", "FloatList"), ("int64_list", "Int64List")),
        start=1,
    ):
        field = feature.field.add()
        field.name = name
        field.number = idx
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = f".tensorflow.{type_name}"
        field.oneof_index = 0

    features = file_desc.message_type.add()
    features.name = "Features"
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True

    key_field = entry.field.add()
    key_field.name = "key"
    key_field.number = 1
    key_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    value_field = entry.field.add()
    value_field.name = "value"
    value_field.number = 2
    value_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    value_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    value_field.type_name = ".tensorflow.Feature"

    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    example = file_desc.message_type.add()
    example.name = "Example"
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


_ExampleMessage = _build_example_message_class()


def _iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"Corrupted TFRecord length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            crc_len = f.read(4)
            if len(crc_len) != 4:
                raise ValueError(f"Corrupted TFRecord length CRC in {path}")
            data = f.read(length)
            if len(data) != length:
                raise ValueError(f"Corrupted TFRecord payload in {path}")
            crc_data = f.read(4)
            if len(crc_data) != 4:
                raise ValueError(f"Corrupted TFRecord data CRC in {path}")
            yield data


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class _Episode:
    actions: np.ndarray
    states: np.ndarray
    prompts: list[str]
    image_bytes: list[bytes]
    wrist_image_bytes: list[bytes]

    @property
    def length(self) -> int:
        return len(self.prompts)


def _parse_episode(raw_example: bytes) -> _Episode:
    example = _ExampleMessage()
    example.ParseFromString(raw_example)
    feature_map = example.features.feature

    num_steps = len(feature_map["steps/is_first"].int64_list.value)
    actions = np.asarray(feature_map["steps/action"].float_list.value, dtype=np.float32).reshape(num_steps, -1)
    states = np.asarray(feature_map["steps/observation/state"].float_list.value, dtype=np.float32).reshape(
        num_steps, -1
    )
    prompts = [value.decode("utf-8") for value in feature_map["steps/language_instruction"].bytes_list.value]
    image_bytes = list(feature_map["steps/observation/image"].bytes_list.value)
    wrist_image_bytes = list(feature_map["steps/observation/wrist_image"].bytes_list.value)

    return _Episode(
        actions=actions,
        states=states,
        prompts=prompts,
        image_bytes=image_bytes,
        wrist_image_bytes=wrist_image_bytes,
    )


@dataclasses.dataclass(frozen=True)
class _DatasetSource:
    config: RLDSDataset
    version_dir: Path
    shard_paths: list[Path]
    num_transitions: int

    def iter_samples(self, action_chunk_size: int, *, shuffle: bool, seed: int, repeat: bool = True) -> Iterator[dict]:
        rng = random.Random(seed)
        while True:
            shard_paths = list(self.shard_paths)
            if shuffle:
                rng.shuffle(shard_paths)
            for shard_path in shard_paths:
                records = list(_iter_tfrecord_records(shard_path))
                if shuffle:
                    rng.shuffle(records)
                for raw_record in records:
                    episode = _parse_episode(raw_record)
                    indices = list(range(episode.length))
                    if shuffle:
                        rng.shuffle(indices)
                    for step_idx in indices:
                        action_indices = np.arange(step_idx, step_idx + action_chunk_size)
                        action_indices = np.minimum(action_indices, episode.length - 1)
                        yield {
                            "image": _decode_image(episode.image_bytes[step_idx]),
                            "wrist_image": _decode_image(episode.wrist_image_bytes[step_idx]),
                            "state": np.asarray(episode.states[step_idx], dtype=np.float32),
                            "actions": np.asarray(episode.actions[action_indices], dtype=np.float32),
                            "prompt": episode.prompts[step_idx],
                        }
            if not repeat:
                return


def _collate_batch(samples: list[dict]) -> dict:
    batch_size = len(samples)
    batch = {
        "image": np.stack([sample["image"] for sample in samples], axis=0),
        "wrist_image": np.stack([sample["wrist_image"] for sample in samples], axis=0),
        "state": np.stack([sample["state"] for sample in samples], axis=0),
        "actions": np.stack([sample["actions"] for sample in samples], axis=0),
        "prompt": np.empty((batch_size,), dtype=object),
    }
    for i, sample in enumerate(samples):
        batch["prompt"][i] = sample["prompt"]
    return batch


class LiberoRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,
        shuffle: bool = True,
        action_chunk_size: int = 10,
        seed: int = 0,
        allow_missing_norm_stats: bool = False,
        repeat: bool = True,
    ):
        if not datasets:
            raise ValueError("At least one RLDS dataset must be configured.")
        weight_sum = sum(dataset.weight for dataset in datasets)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Dataset weights must sum to 1.0, got {weight_sum}.")

        root = Path(data_dir)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._action_chunk_size = action_chunk_size
        self._seed = seed
        self._repeat = repeat
        self._weights = [dataset.weight for dataset in datasets]
        self._sources: list[_DatasetSource] = []

        logging.info(f"Preparing {len(datasets)} LIBERO RLDS datasets...")
        for dataset in datasets:
            version_dir = root / dataset.name / dataset.version
            if not version_dir.exists():
                raise FileNotFoundError(f"RLDS dataset directory not found: {version_dir}")
            shard_paths = sorted(version_dir.glob("*.tfrecord-*"))
            if not shard_paths:
                raise FileNotFoundError(f"No TFRecord shards found in {version_dir}")

            norm_stats_path = version_dir / "norm_stats.json"
            if not norm_stats_path.exists():
                if not allow_missing_norm_stats:
                    raise FileNotFoundError(f"norm_stats.json not found in {version_dir}")
                num_transitions = 0
            else:
                action_stats = json.loads(norm_stats_path.read_text())["norm_stats"]["actions"]
                num_transitions = int(action_stats.get("num_transitions", 0))

            self._sources.append(
                _DatasetSource(
                    config=dataset,
                    version_dir=version_dir,
                    shard_paths=shard_paths,
                    num_transitions=num_transitions,
                )
            )
            transition_count = str(num_transitions) if num_transitions else "unknown"
            logging.info(
                f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f} "
                f"({len(shard_paths)} shards, {transition_count} transitions)"
            )

    def __iter__(self):
        rng = random.Random(self._seed)
        iterators = [
            source.iter_samples(
                self._action_chunk_size,
                shuffle=self._shuffle,
                seed=rng.randint(0, 2**31 - 1),
                repeat=self._repeat,
            )
            for source in self._sources
        ]
        active_indices = list(range(len(iterators)))
        while True:
            samples = []
            for _ in range(self._batch_size):
                while active_indices:
                    active_weights = [self._weights[index] for index in active_indices]
                    source_index = rng.choices(active_indices, weights=active_weights, k=1)[0]
                    try:
                        samples.append(next(iterators[source_index]))
                        break
                    except StopIteration:
                        # Norm-stat computation uses finite sources. Once one
                        # source is exhausted, continue sampling the others.
                        active_indices.remove(source_index)
                if not active_indices and len(samples) < self._batch_size:
                    return
            yield _collate_batch(samples)

    def __len__(self):
        return sum(source.num_transitions for source in self._sources)
