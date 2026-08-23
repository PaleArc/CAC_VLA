# Upstream relationship

This repository is a research fork of
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).

The `upstream` Git remote tracks the official project. PaleARC-specific model,
RLDS, training, and evaluation code is kept in focused modules so upstream
updates can be reviewed separately.

## External repositories

The modified OAT implementation is maintained in a separate fork and will be
pinned at `external/oat` as a Git submodule before publication. The submodule
must point to an immutable commit; generated tokenizer checkpoints are not stored here.

LIBERO-Plus and CALVIN are evaluation dependencies. Their source, datasets, and
assets are not vendored in this repository. Evaluation entry points accept an
explicit repository path or environment variable.

## Local assets

Datasets, model parameters, checkpoints, logs, videos, and virtual environments
must remain outside Git. The project configs use the official pi05 checkpoint by
default and accept `OPENPI_PI05_BASE_PARAMS` for a local or hosted override.
