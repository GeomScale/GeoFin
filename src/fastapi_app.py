"""FastAPI service exposing PorQua least-squares optimization."""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api_core import OptimizationConfig, optimize_least_squares, parse_csv_payload


class ConstraintRequest(BaseModel):
    budget: bool = True
    long_only: bool = True
    upper_bound: float = Field(default=1.0, ge=0.0)


class OptimizationRequest(BaseModel):
    benchmark_column: str = "benchmark"
    solver_name: str = "highs"
    l2_penalty: float = 0.0
    constraints: ConstraintRequest = ConstraintRequest()


app = FastAPI(title="PorQua API", version="0.1.0")


@app.post("/api/v1/optimize")
async def optimize_portfolio(
    data: UploadFile = File(...),
    params: str | None = Form(default=None),
):
    try:
        payload = await data.read()
        dataframe = parse_csv_payload(payload)

        request = OptimizationRequest(**json.loads(params)) if params else OptimizationRequest()
        config = OptimizationConfig(
            benchmark_column=request.benchmark_column,
            solver_name=request.solver_name,
            add_budget=request.constraints.budget,
            long_only=request.constraints.long_only,
            upper_bound=request.constraints.upper_bound,
            l2_penalty=request.l2_penalty,
        )
        return optimize_least_squares(dataframe, config)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
