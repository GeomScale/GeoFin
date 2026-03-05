# PorQua

PorQua is a Python library for portfolio optimization and backtesting.

## FastAPI service (new)

This repository now includes a lightweight FastAPI wrapper around PorQua's
least-squares optimizer so users can upload CSV data and receive optimized
portfolio weights through a REST endpoint.

### Endpoint

- `POST /api/v1/optimize`
- multipart form-data fields:
  - `data` (required): CSV file containing asset return columns and one benchmark column
  - `params` (optional): JSON string with optimization settings

Example `params`:

```json
{
  "benchmark_column": "benchmark",
  "solver_name": "highs",
  "l2_penalty": 0.0,
  "constraints": {
    "budget": true,
    "long_only": true,
    "upper_bound": 0.1
  }
}
```

The response contains optimized weights and key metrics (`objective_value`, `tracking_error`).

### Run locally

```bash
pip install -r requirements-web.txt
PYTHONPATH=src uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t porqua-api .
docker run --rm -p 8000:8000 porqua-api
```
