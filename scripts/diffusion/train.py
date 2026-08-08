#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the state-based Diffusion Policy baseline on a demonstration dataset.

Takes the HDF5 written by ``build_dataset.py``. Pure PyTorch, so pick a GPU with
``CUDA_VISIBLE_DEVICES``.

Usage::

    python scripts/diffusion/train.py \
        --task Dexverse-GraspKettle-v0 \
        --dataset_file datasets/diffusion/Dexverse-GraspKettle-v0.h5 \
        --output_dir runs/dp_Dexverse-GraspKettle-v0
"""

from __future__ import annotations

import argparse

from dexverse.IL.diffusion.config import TrainConfig
from dexverse.IL.diffusion.train import train


def main() -> None:
    defaults = TrainConfig(task="", dataset_files=(), output_dir="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="Task id the demonstrations were recorded on.")
    parser.add_argument("--dataset_file", action="append", dest="dataset_files", required=True,
                        help="Training HDF5 from build_dataset.py. Repeatable.")
    parser.add_argument("--output_dir", required=True, help="Where checkpoints and metrics.json go.")
    parser.add_argument("--obs_horizon", type=int, default=defaults.obs_horizon)
    parser.add_argument("--action_horizon", type=int, default=defaults.action_horizon)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save_every_epochs", type=int, default=defaults.save_every_epochs)
    parser.add_argument("--early_stopping_patience", type=int, default=defaults.early_stopping_patience,
                        help="Epochs without a new best validation loss before stopping; 0 disables.")
    args = parser.parse_args()

    cfg = TrainConfig(**{**vars(args), "dataset_files": tuple(args.dataset_files)})
    print(train(cfg))


if __name__ == "__main__":
    main()
