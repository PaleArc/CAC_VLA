from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
import sys
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pi_oat policy server on the CALVIN benchmark.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
    parser.add_argument(
        "--replan-steps", type=int, default=10, help="Number of predicted actions to execute per query."
    )
    parser.add_argument("--resize-size", type=int, default=224, help="Image resize used before websocket inference.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the original CALVIN dataset root containing validation/.",
    )
    parser.add_argument(
        "--calvin-root",
        required=True,
        help="Path to the local CALVIN repository clone.",
    )
    parser.add_argument(
        "--calvin-config-path",
        default=None,
        help="Optional override for CALVIN config dir. Defaults to <calvin_root>/calvin_models/conf.",
    )
    parser.add_argument(
        "--eval-sequences-path",
        default=None,
        help="Optional path to a JSON file with evaluation sequences. Defaults to CALVIN's generated benchmark set.",
    )
    parser.add_argument("--num-sequences", type=int, default=1000, help="Number of benchmark sequences to evaluate.")
    parser.add_argument("--ep-len", type=int, default=360, help="Max steps per subtask rollout.")
    parser.add_argument("--debug", action="store_true", help="Render the environment while evaluating.")
    parser.add_argument(
        "--disable-egl",
        action="store_true",
        help="Disable CALVIN's EGL renderer and fall back to DIRECT mode.",
    )
    parser.add_argument(
        "--eval-log-dir",
        default="data/calvin/eval",
        help="Directory where CALVIN evaluation summaries will be written.",
    )
    parser.add_argument(
        "--video-out-path",
        default=None,
        help="Optional directory for rollout videos. If omitted, videos are not saved.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of CALVIN evaluation workers.")
    parser.add_argument("--worker-id", type=int, default=0, help="Zero-based worker id for sharded evaluation.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore saved CALVIN progress for this run.")
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge completed worker shard files into the final results.json and exit.",
    )
    parser.add_argument(
        "--run-name",
        default="pi_oat_calvin_eval",
        help="Label used as the results.json key inside the eval log directory.",
    )
    return parser.parse_args()


def _add_calvin_to_syspath(calvin_root: Path) -> Path:
    if not calvin_root.exists():
        raise FileNotFoundError(f"CALVIN repo not found: {calvin_root}")

    calvin_models_root = calvin_root / "calvin_models"
    calvin_env_root = calvin_root / "calvin_env"
    calvin_tacto_root = calvin_env_root / "tacto"
    if not calvin_models_root.exists():
        raise FileNotFoundError(f"Missing CALVIN models directory: {calvin_models_root}")
    if not any(calvin_env_root.iterdir()):
        raise FileNotFoundError(
            f"CALVIN environment directory is empty. Run `git -C {calvin_root} submodule update --init --recursive`."
        )

    # Keep CALVIN's bundled tacto package ahead of the repo root on sys.path.
    for path in (calvin_env_root, calvin_models_root, calvin_tacto_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return calvin_models_root / "conf"


def _load_eval_sequences(eval_sequences_path: str | None, num_sequences: int, get_sequences):
    if eval_sequences_path is None:
        return list(get_sequences(num_sequences))

    payload = json.loads(Path(eval_sequences_path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("eval_sequences", payload.get("sequences", payload))

    if not isinstance(payload, list):
        raise ValueError(f"Unsupported eval sequence payload type: {type(payload)!r}")

    sequences = []
    for item in payload:
        if isinstance(item, dict):
            initial_state = item.get("initial_state")
            sequence = item.get("sequence")
        else:
            initial_state, sequence = item
        if initial_state is None or sequence is None:
            raise ValueError(f"Malformed eval sequence entry: {item!r}")
        sequences.append((initial_state, sequence))
    return sequences[:num_sequences]


def _extract_rgb_static(obs: dict) -> np.ndarray:
    if "rgb_obs" in obs and "rgb_static" in obs["rgb_obs"]:
        return np.asarray(obs["rgb_obs"]["rgb_static"])
    if "rgb_static" in obs:
        return np.asarray(obs["rgb_static"])
    raise KeyError("Could not find CALVIN static RGB image in observation.")


def _extract_rgb_gripper(obs: dict, fallback_shape: tuple[int, ...]) -> np.ndarray:
    if "rgb_obs" in obs and "rgb_gripper" in obs["rgb_obs"]:
        return np.asarray(obs["rgb_obs"]["rgb_gripper"])
    if "rgb_gripper" in obs:
        return np.asarray(obs["rgb_gripper"])
    return np.zeros(fallback_shape, dtype=np.uint8)


def _extract_robot_obs(obs: dict) -> np.ndarray:
    if "robot_obs" in obs:
        return np.asarray(obs["robot_obs"], dtype=np.float32)
    raise KeyError("Could not find CALVIN robot_obs in observation.")


def _make_env(
    dataset_path: Path,
    conf_dir: Path,
    *,
    show_gui: bool,
    disable_egl: bool,
):
    import hydra
    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(dataset_path / ".hydra" / "merged_config.yaml")
    # pi_oat eval only uses the static and gripper RGB streams; dropping the
    # tactile camera avoids extra tacto/pyrender runtime requirements.
    if "tactile" in render_conf.cameras:
        del render_conf.cameras["tactile"]
    if disable_egl:
        render_conf.env.use_egl = False
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)


def _adapt_calvin_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Expected a 7D CALVIN action, got shape {action.shape}.")
    # CALVIN expects a binary gripper command in {-1, 1}.
    action[-1] = 1.0 if action[-1] >= 0 else -1.0
    return action


def _get_progress_path(eval_log_path: Path, run_name: str) -> Path:
    return eval_log_path / f"{run_name}.progress.json"


def _get_done_path(eval_log_path: Path, run_name: str) -> Path:
    return eval_log_path / f"{run_name}.done.json"


def _load_progress(progress_path: Path, run_name: str, expected_num_sequences: int) -> list[int]:
    if not progress_path.exists():
        return []

    payload = json.loads(progress_path.read_text())
    if payload.get("run_name") != run_name:
        LOG.warning("Ignoring progress file with mismatched run_name: %s", progress_path)
        return []

    if payload.get("num_sequences") != expected_num_sequences:
        LOG.warning("Ignoring progress file with mismatched num_sequences: %s", progress_path)
        return []

    results = payload.get("results", [])
    if not isinstance(results, list) or not all(isinstance(item, int) for item in results):
        LOG.warning("Ignoring malformed progress file: %s", progress_path)
        return []

    return results


def _save_progress(progress_path: Path, run_name: str, num_sequences: int, results: list[int]) -> None:
    payload = {
        "run_name": run_name,
        "num_sequences": num_sequences,
        "results": results,
    }
    tmp_path = progress_path.with_suffix(".progress.json.tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(progress_path)


def _save_done(done_path: Path, run_name: str, num_sequences: int, results: list[int]) -> None:
    payload = {
        "run_name": run_name,
        "num_sequences": num_sequences,
        "completed_results": len(results),
    }
    tmp_path = done_path.with_suffix(".done.json.tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(done_path)


def _get_shard_run_name(run_name: str, worker_id: int, num_workers: int) -> str:
    return f"{run_name}.worker_{worker_id:03d}_of_{num_workers:03d}"


def _get_shard_result_path(eval_log_path: Path, run_name: str, worker_id: int, num_workers: int) -> Path:
    return eval_log_path / f"{_get_shard_run_name(run_name, worker_id, num_workers)}.shard.json"


def _save_shard_results(
    shard_path: Path,
    *,
    run_name: str,
    worker_id: int,
    num_workers: int,
    num_sequences: int,
    sequence_indices: list[int],
    results: list[int],
) -> None:
    payload = {
        "run_name": run_name,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "num_sequences": num_sequences,
        "sequence_indices": sequence_indices,
        "results": results,
    }
    tmp_path = shard_path.with_suffix(".shard.json.tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(shard_path)


def _merge_shard_results(
    *,
    eval_log_path: Path,
    run_name: str,
    eval_sequences,
    num_workers: int,
    print_and_save,
) -> list[int]:
    merged_results: list[int | None] = [None] * len(eval_sequences)
    for worker_id in range(num_workers):
        shard_path = _get_shard_result_path(eval_log_path, run_name, worker_id, num_workers)
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing CALVIN eval shard result: {shard_path}")
        payload = json.loads(shard_path.read_text())
        if payload.get("run_name") != run_name:
            raise ValueError(f"Shard has mismatched run_name: {shard_path}")
        if payload.get("worker_id") != worker_id or payload.get("num_workers") != num_workers:
            raise ValueError(f"Shard has mismatched worker metadata: {shard_path}")
        if payload.get("num_sequences") != len(eval_sequences):
            raise ValueError(f"Shard has mismatched num_sequences: {shard_path}")

        sequence_indices = payload.get("sequence_indices", [])
        results = payload.get("results", [])
        if len(sequence_indices) != len(results):
            raise ValueError(f"Shard has mismatched indices/results lengths: {shard_path}")
        for sequence_idx, result in zip(sequence_indices, results, strict=True):
            if not isinstance(sequence_idx, int) or not 0 <= sequence_idx < len(eval_sequences):
                raise ValueError(f"Shard has invalid sequence index {sequence_idx!r}: {shard_path}")
            if merged_results[sequence_idx] is not None:
                raise ValueError(f"Duplicate sequence index {sequence_idx} while merging {shard_path}")
            merged_results[sequence_idx] = int(result)

    missing = [idx for idx, result in enumerate(merged_results) if result is None]
    if missing:
        raise ValueError(f"Missing results for {len(missing)} CALVIN sequences; first missing index: {missing[0]}")

    final_results = [int(result) for result in merged_results]
    print_and_save(final_results, eval_sequences, eval_log_path, epoch=run_name)
    _save_done(_get_done_path(eval_log_path, run_name), run_name, len(eval_sequences), final_results)
    return final_results


class WebsocketCalvinModel:
    def __init__(self, host: str, port: int, replan_steps: int, resize_size: int):
        self._client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        self._replan_steps = replan_steps
        self._resize_size = resize_size
        self._action_plan = collections.deque()

    def reset(self) -> None:
        self._action_plan.clear()

    def step(self, obs: dict, goal: str) -> np.ndarray:
        if not self._action_plan:
            static_img = _extract_rgb_static(obs)
            gripper_img = _extract_rgb_gripper(obs, static_img.shape)
            state = _extract_robot_obs(obs)

            element = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(static_img, self._resize_size, self._resize_size)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(gripper_img, self._resize_size, self._resize_size)
                ),
                "observation/state": state,
                "prompt": str(goal),
            }
            action_chunk = np.asarray(self._client.infer(element)["actions"], dtype=np.float32)
            if len(action_chunk) < self._replan_steps:
                raise ValueError(f"Expected at least {self._replan_steps} predicted actions, got {len(action_chunk)}.")
            self._action_plan.extend(_adapt_calvin_action(action) for action in action_chunk[: self._replan_steps])

        return np.asarray(self._action_plan.popleft(), dtype=np.float32)


def evaluate_policy(
    *,
    model,
    env,
    task_oracle,
    val_annotations,
    eval_sequences,
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
    join_vis_lang,
    get_log_dir,
    debug: bool,
    ep_len: int,
    eval_log_dir: str | None,
    video_out_path: str | None,
    run_name: str,
    sequence_indices: list[int] | None = None,
    write_summary: bool = True,
    resume: bool = True,
) -> list[int]:
    eval_log_path = get_log_dir(eval_log_dir)
    progress_path = _get_progress_path(eval_log_path, run_name)
    done_path = _get_done_path(eval_log_path, run_name)
    if sequence_indices is None:
        sequence_indices = list(range(len(eval_sequences)))
    if len(sequence_indices) != len(eval_sequences):
        raise ValueError("sequence_indices must have the same length as eval_sequences.")

    video_dir = None
    if video_out_path is not None:
        video_dir = Path(video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)

    results = _load_progress(progress_path, run_name, len(eval_sequences)) if resume else []
    start_idx = len(results)

    if start_idx >= len(eval_sequences):
        LOG.info("Evaluation already complete for %s. Reusing %d saved results.", run_name, len(results))
        if write_summary:
            print_and_save(results, eval_sequences, eval_log_path, epoch=run_name)
        _save_done(done_path, run_name, len(eval_sequences), results)
        if progress_path.exists():
            progress_path.unlink()
        return results

    if start_idx > 0:
        LOG.info("Resuming evaluation for %s from sequence %d/%d.", run_name, start_idx, len(eval_sequences))

    remaining_sequence_indices = sequence_indices[start_idx:]
    remaining_eval_sequences = eval_sequences[start_idx:]
    if len(remaining_sequence_indices) != len(remaining_eval_sequences):
        raise ValueError("remaining sequence_indices must have the same length as remaining eval_sequences.")
    remaining_sequences = list(zip(remaining_sequence_indices, remaining_eval_sequences, strict=True))
    progress = remaining_sequences if debug else __import__("tqdm").tqdm(remaining_sequences, position=0, leave=True)
    if not debug and results:
        progress.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )

    for seq_idx, (initial_state, eval_sequence) in progress:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        if debug:
            time.sleep(0.5)
            print()
            print(f"Evaluating sequence {seq_idx}: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")

        frames = []
        for subtask in eval_sequence:
            if debug:
                print(f"{subtask} ", end="")
            obs = env.get_obs()
            lang_annotation = val_annotations[subtask][0]
            model.reset()
            start_info = env.get_info()

            succeeded = False
            for _ in range(ep_len):
                action = model.step(obs, lang_annotation)
                obs, _, _, current_info = env.step(action)
                if video_dir is not None:
                    frames.append(env.render(mode="rgb_array"))
                if debug:
                    join_vis_lang(env.render(mode="rgb_array"), lang_annotation)
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    succeeded = True
                    success_counter += 1
                    break

            if not succeeded:
                if debug:
                    print("fail", end=" ")
                break
            if debug:
                print("success", end=" ")

        results.append(success_counter)
        _save_progress(progress_path, run_name, len(eval_sequences), results)
        if not debug:
            progress.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

        if video_dir is not None and frames:
            suffix = f"len_{success_counter}"
            video_path = video_dir / f"sequence_{seq_idx:04d}_{suffix}.mp4"
            imageio.mimwrite(video_path, [np.asarray(frame) for frame in frames], fps=30)

    if write_summary:
        print_and_save(results, eval_sequences, eval_log_path, epoch=run_name)
    _save_done(done_path, run_name, len(eval_sequences), results)
    if progress_path.exists():
        progress_path.unlink()
    return results


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError(f"--num-workers must be >= 1, got {args.num_workers}.")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError(f"--worker-id must be in [0, {args.num_workers}), got {args.worker_id}.")

    np.random.seed(args.seed)

    calvin_root = Path(args.calvin_root)
    default_conf_dir = _add_calvin_to_syspath(calvin_root)
    conf_dir = Path(args.calvin_config_path) if args.calvin_config_path is not None else default_conf_dir

    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import count_success
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_agent.evaluation.utils import get_log_dir
    from calvin_agent.evaluation.utils import join_vis_lang
    from calvin_agent.evaluation.utils import print_and_save
    import hydra
    from omegaconf import OmegaConf

    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = _load_eval_sequences(args.eval_sequences_path, args.num_sequences, get_sequences)
    eval_log_path = get_log_dir(args.eval_log_dir)

    if args.merge_shards:
        _merge_shard_results(
            eval_log_path=eval_log_path,
            run_name=args.run_name,
            eval_sequences=eval_sequences,
            num_workers=args.num_workers,
            print_and_save=print_and_save,
        )
        return

    model = WebsocketCalvinModel(args.host, args.port, args.replan_steps, args.resize_size)
    val_folder = Path(args.dataset_path) / "validation"
    env = _make_env(val_folder, conf_dir, show_gui=args.debug, disable_egl=args.disable_egl)
    if args.num_workers == 1:
        worker_indices = list(range(len(eval_sequences)))
        worker_sequences = eval_sequences
        worker_run_name = args.run_name
    else:
        worker_indices = [idx for idx in range(len(eval_sequences)) if idx % args.num_workers == args.worker_id]
        worker_sequences = [eval_sequences[idx] for idx in worker_indices]
        worker_run_name = _get_shard_run_name(args.run_name, args.worker_id, args.num_workers)
        LOG.info(
            "Worker %d/%d evaluating %d/%d CALVIN sequences.",
            args.worker_id,
            args.num_workers,
            len(worker_sequences),
            len(eval_sequences),
        )

    results = evaluate_policy(
        model=model,
        env=env,
        task_oracle=task_oracle,
        val_annotations=val_annotations,
        eval_sequences=worker_sequences,
        count_success=count_success,
        get_env_state_for_initial_condition=get_env_state_for_initial_condition,
        print_and_save=print_and_save,
        join_vis_lang=join_vis_lang,
        get_log_dir=get_log_dir,
        debug=args.debug,
        ep_len=args.ep_len,
        eval_log_dir=args.eval_log_dir,
        video_out_path=args.video_out_path,
        run_name=worker_run_name,
        sequence_indices=worker_indices,
        write_summary=args.num_workers == 1,
        resume=not args.no_resume,
    )
    if args.num_workers > 1:
        _save_shard_results(
            _get_shard_result_path(eval_log_path, args.run_name, args.worker_id, args.num_workers),
            run_name=args.run_name,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            num_sequences=len(eval_sequences),
            sequence_indices=worker_indices,
            results=results,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
