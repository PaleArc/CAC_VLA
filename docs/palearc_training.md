# PaleARC training

Use `scripts/run_palearc_training.py` to launch one registered configuration and optionally
override its learning rate, OAT memory dropout, training length, or checkpoint
retention. GPU selection and accelerator settings use the standard JAX
environment variables.

The project configs read these optional environment variables:

- `OPENPI_PI05_BASE_PARAMS`
- `OPENPI_LIBERO_RLDS_DIR`
- `OPENPI_LIBERO_OAT_RLDS_DIR`
- `OPENPI_LIBERO_PLUS_RLDS_DIR`
- `OPENPI_LIBERO_PLUS_OAT_H10_DIR`
- `OPENPI_LIBERO_PLUS_OAT_H20_DIR`
- `OPENPI_LIBERO_PLUS_OAT_H30_DIR`
- `OPENPI_CALVIN_RLDS_DIR`
- `OPENPI_CALVIN_OAT_RLDS_DIR`

Without overrides, dataset paths resolve under `data/rlds/`, which is ignored
by Git.
