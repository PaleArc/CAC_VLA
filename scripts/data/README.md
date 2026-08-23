# OAT data preparation

The three entry points add OAT tokenizer targets to existing RLDS datasets:

- `augment_libero_rlds_with_oat.py`
- `augment_libero_plus_rlds_with_oat.py`
- `augment_calvin_rlds_with_oat.py`

Source and output dataset roots are required CLI arguments. `--oat-root`
defaults to `external/oat`, and `--oat-checkpoint` is always required.
Generated RLDS data and tokenizer checkpoints are not tracked by Git.
