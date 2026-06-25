# -*- coding: utf-8 -*-
"""
参数规模估算脚本 - BioP Causal WorldModel V2.0
用于在不启动完整训练的情况下估算 WorldModel + Actor + Critic
在给定 latent_dim / hidden_dim 下的参数量，以命中 ~30M 的目标预算。

用法:
    python tools/estimate_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.expanded_ncde import (
    WorldModelEncoder,
    WorldModelDecoder,
    ExpandedNCDEFunction,
    ControlledNCDEFunction,
    ActorNetwork,
    CriticNetwork,
)


def _count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def estimate(
    obs_dim: int = 48,
    action_dim: int = 4,
    latent_dim: int = 1024,
    hidden_dim: int = 1536,
    n_ncde_layers: int = 4,
    action_encoder_dim: int = 256,
) -> dict:
    encoder = WorldModelEncoder(obs_dim, latent_dim, hidden_dim)
    decoder = WorldModelDecoder(latent_dim, obs_dim, hidden_dim)
    dynamics = ExpandedNCDEFunction(latent_dim, n_ncde_layers)
    controlled = ControlledNCDEFunction(latent_dim, action_encoder_dim, n_ncde_layers)
    action_encoder = torch.nn.Sequential(
        torch.nn.Linear(action_dim, action_encoder_dim),
        torch.nn.GELU(),
        torch.nn.Linear(action_encoder_dim, action_encoder_dim),
        torch.nn.GELU(),
        torch.nn.Linear(action_encoder_dim, latent_dim),
    )
    actor = ActorNetwork(latent_dim, action_dim, latent_dim)
    critic = CriticNetwork(latent_dim, action_dim, latent_dim)

    target_encoder = WorldModelEncoder(obs_dim, latent_dim, hidden_dim)
    target_decoder = WorldModelDecoder(latent_dim, obs_dim, hidden_dim)
    target_dynamics = ExpandedNCDEFunction(latent_dim, n_ncde_layers)

    components = {
        "encoder": _count(encoder),
        "decoder": _count(decoder),
        "dynamics (autonomous)": _count(dynamics),
        "controlled_dynamics": _count(controlled),
        "action_encoder": _count(action_encoder),
        "actor": _count(actor),
        "critic (q1+q2)": _count(critic),
        "target_encoder": _count(target_encoder),
        "target_decoder": _count(target_decoder),
        "target_dynamics": _count(target_dynamics),
    }
    total = sum(components.values())
    return {"components": components, "total": total}


def main() -> None:
    configs = [
        # (latent_dim, hidden_dim, note)
        (768, 768, "紧凑 ~24M"),
        (896, 768, "小~27M (推荐A)"),
        (896, 896, "中~32M (推荐B,接近30M)"),
        (1024, 768, "默认缩窄~40M"),
        (1024, 1024, "当前默认 ~41M"),
        (1280, 1024, "大 ~63M"),
    ]

    header = (
        f"{'Config':<28} {'Latent':>7} {'Hidden':>7} {'Total':>14}  "
        f"{'vs 30M':>10}"
    )
    print("=" * len(header))
    print("BioP Causal WorldModel - 参数规模估算")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for latent, hidden, note in configs:
        info = estimate(latent_dim=latent, hidden_dim=hidden)
        total = info["total"]
        delta = (total - 30_000_000) / 1_000_000
        marker = "OK" if abs(delta) < 2 else ("+" if delta > 0 else "-")
        print(
            f"{note:<28} {latent:>7} {hidden:>7} {total:>14,}  "
            f"{delta:>+9.2f}M {marker}"
        )

    print("-" * len(header))
    print("结论: 若需严格 ~30M 参数预算，推荐 latent_dim=896, hidden_dim=768 (~27M)，")
    print("      或 latent_dim=896, hidden_dim=896 (~32M)。当前默认 1024/1024 约 41M，")
    print("      略高于预算。调整后请同步修改 models/expanded_ncde.py 中 create_expanded_model")
    print("      默认值，并更新 configs/model_config.yaml。")


if __name__ == "__main__":
    main()
