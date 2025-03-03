# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file


import sys
import os
import unittest
from itertools import product
import pandas as pd
import numpy as np
import scipy
import qpsolvers
from typing import Tuple
import time
import pytest
sys.path.insert(1, 'src')

from src.helper_functions import to_numpy
from src.data_loader import load_data_msci
from src.constraints import Constraints
from src.covariance import Covariance
from src.optimization import *
from src.optimization_data import OptimizationData

"""
    * unit Test class for Constraints class, reach a 100% coverage in this class
"""
class TestConstraintsMethods(unittest.TestCase):
    def test_add_constaints_init(self):
        """
        Test the successful initialization and basic functionality of the Constraints class.
        """
        # Test with valid input
        selection = ["A", "B", "C"]
        constraints = Constraints(selection=selection)
        
        # Test if selection is set correctly
        assert constraints.selection == selection, "Selection should be correctly initialized"
        
        # Test initial values for budget, box, linear, and l1
        assert constraints.budget == {'Amat': None, 'sense': None, 'rhs': None}, "Initial budget should be empty"
        assert constraints.box == {'box_type': 'NA', 'lower': None, 'upper': None}, "Initial box should be empty"
        assert constraints.linear == {'Amat': None, 'sense': None, 'rhs': None}, "Initial linear constraints should be empty"
        assert constraints.l1 == {}, "Initial l1 should be empty"


    def test_add_constraints_init_error(self):
        """
        Test for handling errors during the initialization of the Constraints class.
        """
        # Test with invalid input for selection (non-string elements in selection)
        with pytest.raises(ValueError):
            Constraints(selection=[1, 2, 3])  # Should raise ValueError because selection is not all strings
        
        # Test with empty selection
        with pytest.raises(ValueError):
            Constraints(selection=[])  # Should raise ValueError because empty selection is invalid


    def test_add_budget(self):
        pass # Placeholder for future tests

    def test_add_budget_value_error(self):
        pass 

    def test_add_box(self):
        pass 

    def test_add_box_value_error(self):
        pass 

    def test_add_linear(self):
        pass 

    def test_add_linear_value_error(self):
        pass 

    def test_add_l1(self):
        pass  

    def test_add_l1_type_error(self):
        pass  

    def to_GhAb(self):
        pass 

    def to_GhAb_sense_G_not_None(self):
        pass  

    def to_GhAb_idx__sum_negative(self):
        pass  



class TestQuadraticProgram(unittest.TestCase):

    def __init__(self, testname, universe = 'msci', solver_name = 'cvxopt'):
        super().__init__(testname)
        self._universe = universe
        self._solver_name = solver_name
        self.data = load_data_msci(os.path.join(os.getcwd(), f'data{os.sep}'))

    def test_add_constraints(self):
        universe = self.data['X'].columns
        constraints = Constraints(selection = universe)

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

# --------------------------------------------------------------------------
class TestLeastSquares(TestQuadraticProgram):

    def __init__(self, testname, universe, solver_name, params):
        super().__init__(testname, universe, solver_name)
        self.params = params

    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        self.run_time = time.time() - self.start_time
        recomputed = self.optim.model.objective_value(self.solution.x, False)
        print(f'{self._universe}-{self._solver_name}-{self.params}:\n\t* Found = {self.solution.found}\n\t* Utility = {recomputed}\n\t* Elapsed time: {self.run_time:.3f}(s)')

        self.assertTrue(self.solution.found)

        from_solver = self.solution.obj
        if from_solver is not None:
            self.assertAlmostEqual(from_solver, recomputed)

    def prep_optim(self) -> None:
        selection = self.data['X'].columns

        # Initialize optimization object
        optim = LeastSquares(solver_name = self._solver_name, sparse = True)
        optim.params['l2_penalty'] = self.params.get('l2_penalty', 0)

        # Add constraints
        constraints = Constraints(selection = selection)

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
            constraints.add_l1('turnover', rhs = 1, x0 = dict(zip(selection, np.zeros(selection.size))))

        optim.constraints = constraints

        # Set objective
        optimization_data = OptimizationData(X = self.data['X'], y = self.data['y'], align = True)
        optim.set_objective(optimization_data)

        # Ensure that P and q are numpy arrays
        if 'P' in optim.objective.keys():
            optim.objective['P'] = to_numpy(optim.objective['P'])
        else:
            raise ValueError("Missing matrix 'P' in objective.")

        optim.objective['q'] = to_numpy(optim.objective['q']) if 'q' in optim.objective.keys() else np.zeros(selection.size)

        # Initialize the optimization model
        optim.model_qpsolvers()

        # Attach the optimization object to self
        self.optim = optim

        return None

    def least_square(self):
        self.prep_optim()
        self.optim.solve()
        self.solution = self.optim.model['solution']
        return None



if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(TestQuadraticProgram('test_add_constraints'))

    universes = ['msci']
    solvers = ['highs', 'cvxopt']
    constraint_dict = {'l2_penalty': [0, 1],
                        'add_budget': [True, False],
                        'add_box': ["LongOnly", "LongShort", "Unbounded"],
                        'add_ineq': [True, False],
                        'add_l1': [True, False]}

    constraints_params = list(dict(zip(constraint_dict, x)) for x in product(*constraint_dict.values()))
    tests = product(universes, constraints_params, solvers)
    save_log = {universe : {} for universe in universes}

    for universe, params, solver in tests:
        suite.addTest(TestLeastSquares('least_square', universe, solver, params))

    runner = unittest.TextTestRunner()
    runner.run(suite)
