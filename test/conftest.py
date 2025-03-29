import os
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

# Import your project modules
from src.helper_functions import to_numpy
from src.data_loader import load_data_msci
from src.constraints import Constraints
from src.covariance import Covariance
from src.optimization import *
from src.optimization_data import OptimizationData

@pytest.fixture(scope="session")
def test_data() -> Dict[str, pd.DataFrame]:
    """Load test data once for the entire test session."""
    data = load_data_msci(os.path.join(os.getcwd(), f'data{os.sep}'))
    return data

@pytest.fixture(scope="function")
def sample_universe(test_data) -> pd.Index:
    """Provide a sample universe for testing."""
    return test_data['X'].columns

@pytest.fixture(scope="function")
def constraints(sample_universe) -> Constraints:
    """Create a fresh constraints object for each test."""
    return Constraints(selection=sample_universe)

@pytest.fixture(scope="function")
def optimization_data(test_data) -> OptimizationData:
    """Create optimization data for testing."""
    return OptimizationData(
        X=test_data['X'],
        y=test_data['y'],
        align=True
    )

@pytest.fixture(scope="function")
def least_squares_optimizer(sample_universe, constraints, optimization_data) -> LeastSquares:
    """Create a configured LeastSquares optimizer for testing."""
    optim = LeastSquares(solver_name='cvxopt', sparse=True)
    optim.constraints = constraints
    optim.set_objective(optimization_data)
    
    # Ensure P and q are numpy arrays
    if 'P' in optim.objective:
        optim.objective['P'] = to_numpy(optim.objective['P'])
    optim.objective['q'] = to_numpy(optim.objective.get('q', np.zeros(len(sample_universe))))
    
    optim.model_qpsolvers()
    return optim

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup and teardown for the entire test session."""
    # Setup
    print("\nStarting test session...")
    yield
    # Teardown
    print("\nTest session completed.")

@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    """Track coverage baseline after test session."""
    if exitstatus == 0:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the report for details.") 