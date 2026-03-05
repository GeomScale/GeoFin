"""Core utilities for exposing PorQua optimization through web interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

import numpy as np
import pandas as pd

from constraints import Constraints
from optimization import LeastSquares
from optimization_data import OptimizationData


@dataclass
class OptimizationConfig:
    benchmark_column: str = "benchmark"
    solver_name: str = "highs"
    add_budget: bool = True
    long_only: bool = True
    upper_bound: float = 1.0
    l2_penalty: float = 0.0


def parse_csv_payload(raw_bytes: bytes) -> pd.DataFrame:
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    dataframe = pd.read_csv(io.BytesIO(raw_bytes))
    if dataframe.empty:
        raise ValueError("Uploaded CSV contains no rows.")
    return dataframe


def optimize_least_squares(dataframe: pd.DataFrame, config: OptimizationConfig) -> dict[str, Any]:
    if config.benchmark_column not in dataframe.columns:
        raise ValueError(f"Benchmark column '{config.benchmark_column}' not found.")

    numeric_frame = dataframe.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError("All columns must contain numeric values.")

    feature_columns = [col for col in numeric_frame.columns if col != config.benchmark_column]
    if not feature_columns:
        raise ValueError("CSV must include at least one asset column besides the benchmark column.")

    x = numeric_frame[feature_columns]
    y = numeric_frame[config.benchmark_column]

    optimization_data = OptimizationData(X=x, y=y, align=True)

    constraints = Constraints(selection=x.columns)
    if config.add_budget:
        constraints.add_budget()
    if config.long_only:
        constraints.add_box(box_type="LongOnly", upper=config.upper_bound)

    optimization = LeastSquares(solver_name=config.solver_name, sparse=True)
    optimization.params["l2_penalty"] = config.l2_penalty
    optimization.constraints = constraints
    optimization.set_objective(optimization_data)
    optimization.solve()

    if not optimization.results["status"]:
        raise RuntimeError("Optimization solver did not find a feasible solution.")

    solution = optimization.model["solution"]
    objective_value = optimization.model.objective_value(solution.x, with_const=True)

    return {
        "weights": optimization.results["weights"],
        "metrics": {
            "objective_value": float(objective_value),
            "tracking_error": float(np.sqrt(max(objective_value, 0.0))),
        },
    }
