from fastapi import FastAPI, HTTPException, Query
from data_loader import load_data_msci
from constraints import Constraints
from optimization_data import OptimizationData
from optimization import LeastSquares
import pandas as pd

app = FastAPI(
    title="PorQua GSoC API",
    description="Interactive Portfolio Optimization API for GeomScale",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "PorQua Financial Engine is ready.",
        "endpoints": ["/optimize", "/docs"],
    }


@app.get("/optimize")
def optimize_portfolio(
    n_assets: int = Query(5, ge=1),
    method: str = Query("least_squares"),
):
    """
    Run a single-step portfolio optimization and return weights.

    For now, this uses a Least Squares index replication model on MSCI data.
    """
    try:
        if method != "least_squares":
            raise HTTPException(
                status_code=400,
                detail="Currently only method='least_squares' is supported.",
            )

        # 1. Load data (country indices + world benchmark)
        data = load_data_msci()
        return_series = data["return_series"]
        bm_series = data["bm_series"]

        if n_assets > return_series.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Requested n_assets={n_assets} but only "
                f"{return_series.shape[1]} assets are available.",
            )

        # Restrict to the first n_assets for this example
        selected_assets = return_series.columns[:n_assets]
        X = return_series[selected_assets]

        # 2. Build simple LongOnly, fully-invested constraints
        constraints = Constraints(selection=selected_assets)
        constraints.add_budget(rhs=1.0, sense="=")
        constraints.add_box(box_type="LongOnly")

        # 3. Build optimization data and model
        opt_data = OptimizationData(
            return_series=X,
            bm_series=bm_series,
            align=True,
        )

        optim = LeastSquares(
            solver_name="cvxopt",
            sparse=True,
            verbose=False,
            constraints=constraints,
        )
        optim.set_objective(opt_data)
        optim.solve()

        weights = optim.results["weights"]
        weights_series = pd.Series(weights, index=selected_assets, dtype=float)
        sum_weights = float(weights_series.sum())

        return {
            "status": "success",
            "method": method,
            "universe": {
                "n_assets": len(selected_assets),
                "assets": list(selected_assets),
            },
            "weights": weights,
            "metrics": {
                "sum_weights": sum_weights,
            },
        }

    except HTTPException:
        # Re-raise FastAPI HTTP errors unchanged
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))