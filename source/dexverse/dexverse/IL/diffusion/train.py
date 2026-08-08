# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline DDPM training loop for the diffusion baseline.

Pure PyTorch -- no Isaac Sim, so ``CUDA_VISIBLE_DEVICES`` is the way to pin a
GPU. Writes ``last.pt``, periodic ``epoch_*.pt``, the best-validation ``best.pt``
(holding the EMA weights) and ``metrics.json`` into ``output_dir``.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DatasetConfig, PolicyConfig, TrainConfig
from .dataset import DiffusionDataset
from .model import DiffusionPolicy
from .normalization import normalize, stats_to


def train(cfg: TrainConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset_kwargs = {
        "dataset_files": tuple(cfg.dataset_files),
        "obs_horizon": cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "seed": cfg.seed,
    }
    train_set = DiffusionDataset(DatasetConfig(split="train", **dataset_kwargs))
    val_set = DiffusionDataset(DatasetConfig(split="val", **dataset_kwargs))

    policy_cfg = PolicyConfig(
        state_dim=train_set.state_dim,
        action_dim=train_set.action_dim,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        obs_embed_dim=cfg.obs_embed_dim,
        unet_block_out_channels=tuple(cfg.unet_block_out_channels),
        layers_per_block=cfg.layers_per_block,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
    )
    model = DiffusionPolicy(policy_cfg).to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    scheduler = model.make_noise_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_set) >= cfg.batch_size,
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    stats = stats_to(train_set.stats, device)
    print(
        f"[diffusion] task={cfg.task} state_dim={policy_cfg.state_dim} action_dim={policy_cfg.action_dim} "
        f"episodes={len(train_set.episodes)}/{len(val_set.episodes)} windows={len(train_set)}/{len(val_set)}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    best_val_loss, best_epoch, epochs_since_best = float("inf"), -1, 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            loss = _denoising_loss(model, batch, scheduler, stats, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            _ema_update(ema_model, model, cfg.ema_decay)
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = _validate(ema_model, val_loader, scheduler, stats, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[diffusion] epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        checkpoint = {
            "task": cfg.task,
            "epoch": epoch,
            "policy_cfg": asdict(policy_cfg),
            "train_cfg": asdict(cfg),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "stats": stats_to(train_set.stats, "cpu"),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if epoch % cfg.save_every_epochs == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_since_best = val_loss, epoch, 0
            # Evaluation loads ``model``, so best.pt carries the EMA weights there.
            torch.save({**checkpoint, "model": ema_model.state_dict()}, output_dir / "best.pt")
        else:
            epochs_since_best += 1
            if 0 < cfg.early_stopping_patience <= epochs_since_best:
                print(f"[diffusion] early stop at epoch {epoch}; best was {best_epoch}", flush=True)
                break

    metrics = {
        "task": cfg.task,
        "state_dim": policy_cfg.state_dim,
        "action_dim": policy_cfg.action_dim,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps({"config": asdict(cfg), "policy_cfg": asdict(policy_cfg), "history": history, "metrics": metrics}, indent=2)
    )
    return metrics


def _denoising_loss(model, batch, scheduler, stats, device) -> torch.Tensor:
    obs = normalize(batch["obs"].to(device), stats["state"])
    action = normalize(batch["action"].to(device), stats["action"])
    mask = batch["action_mask"].to(device).unsqueeze(-1)

    noise = torch.randn_like(action)
    timesteps = torch.randint(0, model.cfg.num_train_timesteps, (action.shape[0],), device=device)
    predicted = model(scheduler.add_noise(action, noise, timesteps), timesteps, obs)

    error = F.mse_loss(predicted, noise, reduction="none") * mask
    return error.sum() / mask.expand_as(error).sum().clamp_min(1.0)


@torch.no_grad()
def _validate(model, loader, scheduler, stats, device) -> float:
    model.eval()
    losses = [_denoising_loss(model, batch, scheduler, stats, device).item() for batch in loader]
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _ema_update(ema_model, model, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
