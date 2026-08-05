"""Evaluate coarse tensor fire forecasts against canonical teacher traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from aeolus.config import load_config
from aeolus.core.simulator import AeolusSimulator
from aeolus.core.state import FirePhase
from aeolus.envs.tensor_incident import TensorIncidentEnv


def canonical_metrics(simulator: AeolusSimulator) -> dict[str, float]:
    truth = simulator.state.truth
    burnable = max(float((~truth.barrier).sum()), 1.0)
    value_sum = max(float(truth.asset_value.sum()), 1.0)
    burning = truth.phase == int(FirePhase.FLAMING)
    burned = truth.phase == int(FirePhase.BURNED)
    return {
        "burning_fraction": float(burning.sum() / burnable),
        "burned_fraction": float(burned.sum() / burnable),
        "cumulative_fire_fraction": float((burning | burned).sum() / burnable),
        "expected_loss": float(
            ((burned.astype(np.float32) + 0.5 * burning) * truth.asset_value).sum() / value_sum
        ),
    }


def ensemble_summary(value: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(value, torch.tensor([0.05, 0.5, 0.95], device=value.device))
    return {
        "mean": float(value.mean()),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def one_case(
    config,
    *,
    seed: int,
    warmup_steps: int,
    forecast_steps: int,
    ensemble_size: int,
    grid_size: int,
    segments: int,
    fire_substeps: int,
    ignition_rate_multiplier: float,
    device: torch.device,
) -> dict[str, object]:
    simulator = AeolusSimulator(config)
    simulator.reset(seed)
    hold = {resource_id: 0 for resource_id in simulator.agent_ids}
    for _ in range(warmup_steps):
        simulator.decision_step(hold)
    canonical_initial = canonical_metrics(simulator)

    surrogate = TensorIncidentEnv(
        config,
        batch_size=ensemble_size,
        max_segments=segments,
        grid_size=grid_size,
        fire_substeps=fire_substeps,
        device=device,
        terminate_on_completion=False,
        terminate_on_escape=False,
    )
    surrogate.reset(seed=seed + 1_000_003)
    surrogate.base_ignition_rate_min *= ignition_rate_multiplier
    surrogate.initialize_fire_from_canonical(simulator)
    tensor_initial_loss, _, _ = surrogate._outcome_metrics(surrogate.state)
    tensor_burnable = (~surrogate.state.barrier).to(surrogate.dtype).sum(dim=(1, 2)).clamp_min(1.0)
    tensor_initial_cumulative = (surrogate.state.burning + surrogate.state.burned).sum(
        dim=(1, 2)
    ) / tensor_burnable

    actions = torch.zeros(
        (ensemble_size, surrogate.num_resources),
        device=device,
        dtype=torch.long,
    )
    transition = None
    for _ in range(forecast_steps):
        simulator.decision_step(hold)
        transition = surrogate.step(actions)
    assert transition is not None
    canonical_final = canonical_metrics(simulator)
    canonical_delta_loss = canonical_final["expected_loss"] - canonical_initial["expected_loss"]
    canonical_delta_cumulative = (
        canonical_final["cumulative_fire_fraction"] - canonical_initial["cumulative_fire_fraction"]
    )
    tensor_delta_loss = transition.expected_loss - tensor_initial_loss
    tensor_final_cumulative = (surrogate.state.burning + surrogate.state.burned).sum(
        dim=(1, 2)
    ) / tensor_burnable
    tensor_delta_cumulative = tensor_final_cumulative - tensor_initial_cumulative
    loss_summary = ensemble_summary(tensor_delta_loss)
    cumulative_summary = ensemble_summary(tensor_delta_cumulative)
    return {
        "seed": seed,
        "start_minute": warmup_steps * config.decision_interval_min,
        "forecast_minutes": forecast_steps * config.decision_interval_min,
        "ignition_rate_multiplier": ignition_rate_multiplier,
        "canonical_initial": canonical_initial,
        "canonical_final": canonical_final,
        "canonical_delta_expected_loss": canonical_delta_loss,
        "canonical_delta_cumulative_fire_fraction": canonical_delta_cumulative,
        "tensor_delta_expected_loss": loss_summary,
        "tensor_delta_cumulative_fire_fraction": cumulative_summary,
        "absolute_error_delta_expected_loss": abs(loss_summary["mean"] - canonical_delta_loss),
        "absolute_error_delta_cumulative_fire_fraction": abs(
            cumulative_summary["mean"] - canonical_delta_cumulative
        ),
        "canonical_loss_in_tensor_90pct_interval": (
            loss_summary["p05"] <= canonical_delta_loss <= loss_summary["p95"]
        ),
        "canonical_cumulative_in_tensor_90pct_interval": (
            cumulative_summary["p05"] <= canonical_delta_cumulative <= cumulative_summary["p95"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cluster_tensor_incident.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9103)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--forecast-steps", type=int, default=12)
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--fire-substeps", type=int, default=2)
    parser.add_argument("--ignition-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--out",
        default="results/tensor_incident/canonical_teacher_check.json",
    )
    args = parser.parse_args()
    experiment = load_config(args.config)
    config = replace(experiment.scenario, terminate_on_escape=False)
    device = torch.device(args.device)
    cases = [
        one_case(
            config,
            seed=args.seed + index * 1009,
            warmup_steps=args.warmup_steps,
            forecast_steps=args.forecast_steps,
            ensemble_size=args.ensemble_size,
            grid_size=args.grid_size,
            segments=args.segments,
            fire_substeps=args.fire_substeps,
            ignition_rate_multiplier=args.ignition_rate_multiplier,
            device=device,
        )
        for index in range(args.cases)
    ]
    mean_loss_error = float(np.mean([case["absolute_error_delta_expected_loss"] for case in cases]))
    mean_cumulative_error = float(
        np.mean([case["absolute_error_delta_cumulative_fire_fraction"] for case in cases])
    )
    results = {
        "schema_version": 1,
        "status": "diagnostic_not_calibrated",
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "mean_absolute_error_delta_expected_loss": mean_loss_error,
            "mean_absolute_error_delta_cumulative_fire_fraction": (mean_cumulative_error),
            "loss_interval_coverage": float(
                np.mean([case["canonical_loss_in_tensor_90pct_interval"] for case in cases])
            ),
            "cumulative_fire_interval_coverage": float(
                np.mean([case["canonical_cumulative_in_tensor_90pct_interval"] for case in cases])
            ),
        },
        "interpretation": (
            "This projects canonical snapshots into the coarse tensor state and "
            "checks uncontrolled forecast deltas. It is an initial teacher "
            "diagnostic, not a calibrated suppression or historical-validity result."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
