import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# Ensure 'src' modules (like data_loader) are importable when src.api is loaded
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.api import app


client = TestClient(app)


class TestAPI(unittest.TestCase):
    def test_home_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "online")
        self.assertIn("/optimize", payload["endpoints"])

    @patch("src.api.LeastSquares")
    @patch("src.api.load_data_msci")
    def test_optimize_least_squares(self, mock_load_data_msci, mock_least_squares):
        # Arrange: mock data loader to avoid filesystem dependency
        import pandas as pd

        return_series = pd.DataFrame(
            {"US": [0.01, 0.02], "UK": [0.03, 0.04], "DE": [0.01, -0.01]}
        )
        bm_series = pd.DataFrame({"NDDLWI": [0.02, 0.01]})
        mock_load_data_msci.return_value = {
            "return_series": return_series,
            "bm_series": bm_series,
        }

        # Arrange: mock optimizer to avoid calling qpsolvers
        mock_instance = MagicMock()
        mock_instance.results = {
            "weights": {"US": 0.6, "UK": 0.3, "DE": 0.1},
            "status": True,
        }
        mock_least_squares.return_value = mock_instance

        # Act
        response = client.get("/optimize?n_assets=3&method=least_squares")

        # Assert
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["method"], "least_squares")
        self.assertEqual(payload["universe"]["n_assets"], 3)
        self.assertCountEqual(
            payload["universe"]["assets"], ["US", "UK", "DE"]
        )
        self.assertAlmostEqual(payload["metrics"]["sum_weights"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

