import dataclasses

import train as _train
import tyro

import openpi.training.config as _config
import openpi.training.optimizer as _optimizer


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    exp_name: str
    peak_lr: float | None = None
    oat_expert_memory_dropout_rate: float | None = None
    num_train_steps: int | None = None
    save_interval: int | None = None
    keep_period: int | None = None
    project_name: str = "palearc"
    resume: bool = True
    overwrite: bool = False


def build_config(args: Args) -> _config.TrainConfig:
    config = _config.get_config(args.config_name)
    model = config.model
    lr_schedule = config.lr_schedule

    if args.peak_lr is not None:
        if not isinstance(lr_schedule, _optimizer.CosineDecaySchedule):
            raise TypeError(f"{config.name} does not use a CosineDecaySchedule.")
        lr_schedule = dataclasses.replace(
            lr_schedule,
            peak_lr=args.peak_lr,
            decay_lr=args.peak_lr,
        )

    if args.oat_expert_memory_dropout_rate is not None:
        if not hasattr(model, "oat_expert_memory_dropout_rate"):
            raise TypeError(f"{config.name} is not an OAT model config.")
        model = dataclasses.replace(
            model,
            oat_expert_memory_dropout_rate=args.oat_expert_memory_dropout_rate,
        )

    return dataclasses.replace(
        config,
        model=model,
        lr_schedule=lr_schedule,
        exp_name=args.exp_name,
        project_name=args.project_name,
        num_train_steps=args.num_train_steps or config.num_train_steps,
        save_interval=args.save_interval or config.save_interval,
        keep_period=args.keep_period if args.keep_period is not None else config.keep_period,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def main(args: Args) -> None:
    _train.main(build_config(args))


if __name__ == "__main__":
    main(tyro.cli(Args))
