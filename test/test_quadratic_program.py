import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from typing import Dict, Any

from src.constraints import Constraints
from src.optimization import LeastSquares
from src.optimization_data import OptimizationData

@pytest.mark.unit
class TestConstraints:
    """Unit tests for the Constraints class."""
    
    @given(st.data())
    def test_add_budget_constraint(self, data, constraints):
        """Test adding budget constraint with property-based testing."""
        constraints.add_budget()
        GhAb = constraints.to_GhAb()
        
        assert GhAb['G'].shape == (1, len(constraints.selection))
        assert GhAb['h'].shape == (1,)
        assert GhAb['A'].shape == (0, len(constraints.selection))
        assert GhAb['b'].shape == (0,)
        
        # Verify budget constraint sums to 1
        assert np.isclose(np.sum(GhAb['G']), 1.0)
        assert np.isclose(GhAb['h'][0], 1.0)

    @pytest.mark.parametrize("box_type", ["LongOnly", "LongShort", "Unbounded"])
    def test_add_box_constraints(self, constraints, box_type):
        """Test adding box constraints with different types."""
        constraints.add_box(box_type)
        GhAb = constraints.to_GhAb(True)
        
        n = len(constraints.selection)
        if box_type == "LongOnly":
            expected_G_rows = 2 * n + 1  # Budget + 2 box constraints per asset
        elif box_type == "LongShort":
            expected_G_rows = 2 * n + 1  # Budget + 2 box constraints per asset
        else:  # Unbounded
            expected_G_rows = 1  # Only budget constraint
            
        assert GhAb['G'].shape == (expected_G_rows, n)
        assert GhAb['h'].shape == (expected_G_rows,)

    @given(st.data())
    def test_add_linear_constraints(self, data, constraints):
        """Test adding linear constraints with property-based testing."""
        n = len(constraints.selection)
        k = data.draw(st.integers(min_value=1, max_value=5))
        
        # Generate random linear constraints
        A = pd.DataFrame(
            np.random.randn(k, n),
            columns=constraints.selection
        )
        sense = pd.Series(['='] * k)
        rhs = pd.Series(np.random.randn(k))
        
        constraints.add_linear(A, None, sense, rhs, None)
        GhAb = constraints.to_GhAb()
        
        assert GhAb['A'].shape == (k, n)
        assert GhAb['b'].shape == (k,)

@pytest.mark.integration
class TestLeastSquaresOptimization:
    """Integration tests for the LeastSquares optimization."""
    
    @pytest.mark.parametrize("solver", ["cvxopt", "highs"])
    def test_optimization_solution(self, least_squares_optimizer, solver):
        """Test optimization solution with different solvers."""
        least_squares_optimizer.solver_name = solver
        least_squares_optimizer.solve()
        solution = least_squares_optimizer.model['solution']
        
        assert solution.found
        assert solution.x is not None
        assert len(solution.x) == len(least_squares_optimizer.constraints.selection)
        
        # Verify constraints are satisfied
        GhAb = least_squares_optimizer.constraints.to_GhAb(True)
        G, h = GhAb['G'], GhAb['h']
        A, b = GhAb['A'], GhAb['b']
        
        # Check inequality constraints
        if G.size > 0:
            assert np.all(G @ solution.x <= h)
        
        # Check equality constraints
        if A.size > 0:
            assert np.allclose(A @ solution.x, b)

    @pytest.mark.performance
    def test_optimization_performance(self, least_squares_optimizer, benchmark):
        """Benchmark optimization performance."""
        def run_optimization():
            least_squares_optimizer.solve()
            return least_squares_optimizer.model['solution']
        
        result = benchmark(run_optimization)
        assert result.found
        assert result.stats.total < 1.0  # Should complete within 1 second

@pytest.mark.property
class TestOptimizationProperties:
    """Property-based tests for optimization properties."""
    
    @given(st.data())
    def test_objective_value_consistency(self, data, least_squares_optimizer):
        """Test that objective value is consistent with solution."""
        least_squares_optimizer.solve()
        solution = least_squares_optimizer.model['solution']
        
        # Compute objective value directly
        x = solution.x
        P = least_squares_optimizer.objective['P']
        q = least_squares_optimizer.objective['q']
        
        direct_value = 0.5 * x.T @ P @ x + q.T @ x
        
        # Compare with solver's reported value
        assert np.isclose(direct_value, solution.obj, rtol=1e-6)

    @given(st.data())
    def test_solution_scaling(self, data, least_squares_optimizer):
        """Test that solution scales appropriately with problem size."""
        n = len(least_squares_optimizer.constraints.selection)
        scale = data.draw(st.floats(min_value=0.1, max_value=10.0))
        
        # Scale the objective
        least_squares_optimizer.objective['P'] *= scale
        least_squares_optimizer.objective['q'] *= scale
        
        least_squares_optimizer.solve()
        scaled_solution = least_squares_optimizer.model['solution']
        
        # Solution should be invariant under objective scaling
        assert np.allclose(scaled_solution.x, least_squares_optimizer.model['solution'].x) 