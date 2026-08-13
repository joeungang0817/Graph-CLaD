"""Reusable Phase 0 smoke run for the supplied CLaD latent dynamics model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, int] = {
    "batch_size": 2,
    "history_length": 2,
    "num_views": 2,
    "proprio_dim": 6,
    "visual_dim_per_view": 64,
    "vl_dim": 64,
    "hidden_dim": 64,
    "action_dim": 5,
}


def normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    if isinstance(raw.get("smoke_test"), Mapping):
        raw = dict(raw["smoke_test"])
    if "semantic_feature_dim" in raw and "vl_dim" not in raw:
        raw["vl_dim"] = raw.pop("semantic_feature_dim")
    return {key: raw[key] for key in DEFAULT_CONFIG if key in raw}


def run_smoke(config: Mapping[str, Any] | None = None, device: str = "cpu") -> dict[str, Any]:
    """Run train, backward, evaluation, and EMA checks without changing baseline code."""

    import torch
    from unittest.mock import patch

    from baseline_code.Attentions import CrossAttnBlock
    from baseline_code.LatentDynamics import LatentDynamics

    cfg = {**DEFAULT_CONFIG, **normalize_config(config)}
    if cfg["num_views"] * cfg["visual_dim_per_view"] != 2 * cfg["vl_dim"]:
        raise ValueError("flattened visual features must equal 2 * vl_dim")
    if cfg["vl_dim"] != cfg["hidden_dim"]:
        raise ValueError("vl_dim must equal hidden_dim for the supplied baseline")

    torch.manual_seed(0)
    model = LatentDynamics(
        proprio_dim=cfg["proprio_dim"],
        vl_dim=cfg["vl_dim"],
        hidden_dim=cfg["hidden_dim"],
        action_dim=cfg["action_dim"],
    ).to(device)
    inputs = {
        "v_history": torch.randn(
            cfg["batch_size"], cfg["history_length"], cfg["num_views"],
            cfg["visual_dim_per_view"], device=device,
        ),
        "p_history": torch.randn(
            cfg["batch_size"], cfg["history_length"], cfg["proprio_dim"], device=device,
        ),
        "prev_action": torch.randn(cfg["batch_size"], cfg["action_dim"], device=device),
        "lang": torch.randn(cfg["batch_size"], cfg["vl_dim"], device=device),
        "p_next": torch.randn(cfg["batch_size"], cfg["proprio_dim"], device=device),
        "v_next": torch.randn(
            cfg["batch_size"], cfg["num_views"], cfg["visual_dim_per_view"], device=device,
        ),
    }
    original_forward = CrossAttnBlock.forward

    def attention_forward(self, x, y, return_attn_weights=False, prefer_flash=True):
        return original_forward(
            self, x, y, return_attn_weights=return_attn_weights,
            prefer_flash=prefer_flash if torch.cuda.is_available() else False,
        )

    model.train()
    with patch.object(CrossAttnBlock, "forward", attention_forward):
        losses = model(**inputs)
    total = sum(losses.values())
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("Phase 0 backward produced missing or non-finite gradients")

    model.eval()
    with torch.no_grad(), patch.object(CrossAttnBlock, "forward", attention_forward):
        pred_p, pred_s = model(
            inputs["v_history"], inputs["p_history"], inputs["prev_action"], inputs["lang"]
        )
    expected = (cfg["batch_size"], cfg["hidden_dim"])
    if tuple(pred_p.shape) != expected or tuple(pred_s.shape) != expected:
        raise RuntimeError("Phase 0 evaluation output shape mismatch")

    online = next(model.p_backbone.parameters())
    target = next(model.p_backbone_target.parameters())
    with torch.no_grad():
        model.update_ema()
        ema_initialized = torch.allclose(target, online)

    return {
        "status": "pass",
        "device": device,
        "config": cfg,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "prediction_shape": list(expected),
        "finite_gradients": True,
        "ema_initialized": bool(ema_initialized),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    result = run_smoke(config=config, device=args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
