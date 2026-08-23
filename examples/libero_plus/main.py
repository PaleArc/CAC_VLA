from __future__ import annotations

# ruff: noqa: E402
import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import sys


def _bootstrap_libero_plus() -> tuple[pathlib.Path, pathlib.Path]:
    project_root = pathlib.Path(__file__).resolve().parents[2]
    default_libero_plus_root = project_root / "external" / "libero-plus"
    libero_plus_root = pathlib.Path(os.environ.get("LIBERO_PLUS_ROOT", str(default_libero_plus_root))).resolve()
    if not libero_plus_root.exists():
        raise FileNotFoundError(
            f"LIBERO-plus root not found: {libero_plus_root}. Set LIBERO_PLUS_ROOT to the cloned LIBERO-plus repo."
        )

    sys.path.insert(0, str(libero_plus_root))

    config_dir = pathlib.Path(
        os.environ.get("LIBERO_CONFIG_PATH", str(project_root / "data" / "libero_plus" / "config"))
    ).resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = libero_plus_root / "libero" / "libero"
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config_file.write_text(
            "\n".join(
                [
                    f"benchmark_root: {benchmark_root}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    f"datasets: {libero_plus_root / 'libero' / 'datasets'}",
                    f"assets: {benchmark_root / 'assets'}",
                    "",
                ]
            )
        )

    return libero_plus_root, config_file


LIBERO_PLUS_ROOT, LIBERO_PLUS_CONFIG = _bootstrap_libero_plus()

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_PLUS_CATEGORY_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO-plus environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"
    task_order_index: int = 0
    max_tasks: int | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"
    result_out_path: str = ""
    save_videos: bool = True
    worker_id: int = 0
    num_workers: int = 1
    seed: int = 7


def eval_libero_plus(args: Args) -> None:
    if args.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}.")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError(f"worker_id must be in [0, {args.num_workers}), got {args.worker_id}.")

    np.random.seed(args.seed)
    _check_libero_plus_paths()

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Options: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[args.task_suite_name](task_order_index=args.task_order_index)
    num_tasks_in_suite = task_suite.n_tasks
    num_tasks_to_run = num_tasks_in_suite if args.max_tasks is None else min(args.max_tasks, num_tasks_in_suite)
    task_metadata = _load_libero_plus_task_metadata(args.task_suite_name)
    logging.info(f"LIBERO-plus root: {LIBERO_PLUS_ROOT}")
    logging.info(f"LIBERO config: {LIBERO_PLUS_CONFIG}")
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Tasks: {num_tasks_to_run}/{num_tasks_in_suite}")

    episodes_by_task: dict[int, list[int]] = collections.defaultdict(list)
    for task_id in range(num_tasks_to_run):
        for episode_idx in range(args.num_trials_per_task):
            global_episode_idx = task_id * args.num_trials_per_task + episode_idx
            if global_episode_idx % args.num_workers == args.worker_id:
                episodes_by_task[task_id].append(episode_idx)
    assigned_task_ids = sorted(episodes_by_task)
    assigned_episodes = sum(len(episode_indices) for episode_indices in episodes_by_task.values())
    logging.info(
        "Worker %d/%d evaluating %d/%d tasks and %d/%d episodes.",
        args.worker_id,
        args.num_workers,
        len(assigned_task_ids),
        num_tasks_to_run,
        assigned_episodes,
        num_tasks_to_run * args.num_trials_per_task,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    task_results = []
    category_totals = {category: {"episodes": 0, "successes": 0} for category in LIBERO_PLUS_CATEGORY_ORDER}
    for task_id in tqdm.tqdm(assigned_task_ids):
        task = task_suite.get_task(task_id)
        metadata = task_metadata.get(task.name, {})
        task_category = metadata.get("category", "Unknown")
        task_difficulty_level = metadata.get("difficulty_level")
        if task_category not in category_totals:
            category_totals[task_category] = {"episodes": 0, "successes": 0}
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(episodes_by_task[task_id]):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[min(episode_idx, len(initial_states) - 1)])

            t = 0
            done = False
            replay_images = []

            logging.info(
                f"Starting episode {task_episodes + 1} "
                f"(task_id={task_id}, episode_idx={episode_idx}, worker={args.worker_id}/{args.num_workers})..."
            )
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan_steps={args.replan_steps}, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.exception(f"Caught exception while evaluating task_id={task_id}: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            category_totals[task_category]["episodes"] += 1
            if done:
                category_totals[task_category]["successes"] += 1

            suffix = "success" if done else "failure"
            if args.save_videos and replay_images:
                task_hash = hashlib.sha1(str(task_description).encode("utf-8")).hexdigest()[:8]
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_task{task_id:04d}_episode{episode_idx:03d}_{suffix}_{task_hash}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes else float("nan")
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else float("nan")
        task_results.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task_description,
                "category": task_category,
                "difficulty_level": task_difficulty_level,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes else float("nan")
    category_results = {
        category: {
            "episodes": counts["episodes"],
            "successes": counts["successes"],
            "success_rate": (
                float(counts["successes"]) / float(counts["episodes"]) if counts["episodes"] else float("nan")
            ),
        }
        for category, counts in category_totals.items()
    }
    logging.info(f"Total success rate: {final_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    for category in LIBERO_PLUS_CATEGORY_ORDER:
        counts = category_results.get(category, {"episodes": 0, "successes": 0, "success_rate": float("nan")})
        logging.info(
            "Category %s success rate: %s (%s/%s)",
            category,
            counts["success_rate"],
            counts["successes"],
            counts["episodes"],
        )

    if args.result_out_path:
        result_path = pathlib.Path(args.result_out_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as f:
            json.dump(
                {
                    "task_suite_name": args.task_suite_name,
                    "task_order_index": args.task_order_index,
                    "worker_id": args.worker_id,
                    "num_workers": args.num_workers,
                    "num_trials_per_task": args.num_trials_per_task,
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "total_success_rate": final_success_rate,
                    "category_results": category_results,
                    "tasks": task_results,
                },
                f,
                indent=2,
                allow_nan=True,
            )


def _load_libero_plus_task_metadata(task_suite_name: str) -> dict[str, dict[str, object]]:
    classification_path = LIBERO_PLUS_ROOT / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not classification_path.exists():
        logging.warning("LIBERO-plus task classification not found: %s", classification_path)
        return {}

    with classification_path.open("r") as f:
        classifications = json.load(f)

    return {row["name"]: row for row in classifications.get(task_suite_name, [])}


def _check_libero_plus_paths() -> None:
    paths = {
        "bddl_files": pathlib.Path(get_libero_path("bddl_files")),
        "init_states": pathlib.Path(get_libero_path("init_states")),
        "assets": pathlib.Path(get_libero_path("assets")),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LIBERO-plus paths:\n" + "\n".join(missing))


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_plus)
