import sys
import os
import unittest
import pandas as pd
import numpy as np
import time
from itertools import product
from typing import Tuple

sys.path.insert(1, 'src')

from helper_functions import to_numpy
from data_loader import load_data_msci
from constraints import Constraints
from optimization import LeastSquares
from optimization_data import OptimizationData


class TestQuadraticProgram(unittest.TestCase):

    def setUp(self):
        """Setup method to load data before running tests."""
        self._universe = 'msci'
        self._solver_name = 'cvxopt'
        data_path = os.path.abspath(os.path.join(os.getcwd(), f'data{os.sep}'))
        self.data = load_data_msci(data_path)

    def test_add_constraints(self):
        """Test the addition of constraints to an optimization problem."""
        universe = self.data['X'].columns
        constraints = Constraints(selection=universe)

        constraints.add_budget()
        constraints.add_box("LongOnly")

        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '<=', 1)
        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '>=', -1)
        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '=', 0.5)

        sub_universe = universe[:universe.size // 2]
        linear_constraints = pd.DataFrame(np.random.rand(3, sub_universe.size), columns=sub_universe)
        sense = pd.Series(np.repeat('=', 3))
        rhs = pd.Series(np.ones(3))
        constraints.add_linear(linear_constraints, None, sense, rhs, None)

        GhAb = constraints.to_GhAb()

        self.assertEqual(GhAb['G'].shape, (2, universe.size))
        self.assertEqual(GhAb['h'].shape, (2,))
        self.assertEqual(GhAb['A'].shape, (5, universe.size))
        self.assertEqual(GhAb['b'].shape, (5,))

        GhAb_with_box = constraints.to_GhAb(True)

        self.assertEqual(GhAb_with_box['G'].shape, (2 + 2 * universe.size, universe.size))
        self.assertEqual(GhAb_with_box['h'].shape, (2 + 2 * universe.size,))
        self.assertEqual(GhAb_with_box['A'].shape, (5, universe.size))
        self.assertEqual(GhAb_with_box['b'].shape, (5,))


class TestLeastSquares(TestQuadraticProgram):

    def setUp(self):
        """Setup test environment before running each test."""
        super().setUp()
        self.params = {'l2_penalty': 0, 'add_budget': True, 'add_box': "LongOnly"}
        self.start_time = time.time()

    def tearDown(self):
        """Tear down and log execution time after test."""
        self.run_time = time.time() - self.start_time
        recomputed = self.optim.model.objective_value(self.solution.x, False)

        print(f'{self._universe}-{self._solver_name}-{self.params}:\n'
              f'\t* Found = {self.solution.found}\n'
              f'\t* Utility = {recomputed}\n'
              f'\t* Elapsed time: {self.run_time:.3f}(s)')

        self.assertTrue(self.solution.found)

        if self.solution.obj is not None:
            self.assertAlmostEqual(self.solution.obj, recomputed)

    def prep_optim(self):
        """Prepare the optimization model."""
        selection = self.data['X'].columns

        optim = LeastSquares(solver_name=self._solver_name, sparse=True)
        optim.params['l2_penalty'] = self.params.get('l2_penalty', 0)

        constraints = Constraints(selection=selection)

        if self.params.get('add_budget', False):
            constraints.add_budget()
        if self.params.get('add_box', None) is not None:
            constraints.add_box(self.params.get('add_box'))
        if self.params.get('add_ineq', False):
            linear_constraints = pd.DataFrame(np.random.rand(3, selection.size), columns=selection)
            sense = pd.Series(np.repeat('<=', 3))
            rhs = pd.Series(np.full(3, 0.5))
            constraints.add_linear(linear_constraints, None, sense, rhs, None)
        if self.params.get('add_l1', False):
            constraints.add_l1('turnover', rhs=1, x0=dict(zip(selection, np.zeros(selection.size))))

        optim.constraints = constraints

        optimization_data = OptimizationData(X=self.data['X'], y=self.data['y'], align=True)
        optim.set_objective(optimization_data)

        if 'P' in optim.objective.keys():
            optim.objective['P'] = to_numpy(optim.objective['P'])
        else:
            raise ValueError("Missing matrix 'P' in objective.")

        optim.objective['q'] = to_numpy(optim.objective['q']) if 'q' in optim.objective.keys() else np.zeros(selection.size)

        optim.model_qpsolvers()

        self.optim = optim

    def test_least_square(self):
        """Run least squares optimization test."""
        self.prep_optim()
        self.optim.solve()
        self.solution = self.optim.model['solution']
        self.assertTrue(self.solution.found)


if __name__ == '__main__':
    unittest.main()
