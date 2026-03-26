@echo off
conda run --live-stream -n multifit-optveri python scripts\run_obv.py plan --config configs\experiments\paper_base.toml %*
