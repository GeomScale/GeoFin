import unittest
import numpy as np
from qp_problems import LinearProgram

class TestLinearProgram(unittest.TestCase):
    def setUp(self):
        """Initialize a LinearProgram instance before each test."""
        self.lp = LinearProgram(params={'solver_name': 'some_solver'})
    
    def test_set_objective(self):
        """Test setting the objective function."""
        q = np.array([1, 2])
        self.lp.set_objective(q)
        self.assertTrue(hasattr(self.lp, 'q'))
        np.testing.assert_array_equal(self.lp.q, q)
    
    def test_add_variables(self):
        """Test adding variables."""
        self.lp.add_variable('x1', 0, 10)
        self.lp.add_variable('x2', 0, 5)
        self.assertEqual(len(self.lp.variables), 2)
    
    def test_add_constraints(self):
        """Test adding constraints."""
        A = np.array([[1, 1], [2, 1]])
        b = np.array([5, 8])
        self.lp.add_constraint(A, b, '<=')
        self.assertEqual(len(self.lp.constraints), 2)
    
    def test_solve(self):
        """Test solving the linear program."""
        self.lp.set_objective(np.array([1, 2]))
        self.lp.add_variable('x1', 0, 10)
        self.lp.add_variable('x2', 0, 5)
        self.lp.add_constraint(np.array([[1, 1], [2, 1]]), np.array([5, 8]), '<=')
        
        result = self.lp.solve()
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(self.lp, 'solution'))
    
    def test_objective_value(self):
        """Test retrieving the objective value after solving."""
        self.lp.set_objective(np.array([1, 2]))
        self.lp.add_variable('x1', 0, 10)
        self.lp.add_variable('x2', 0, 5)
        self.lp.add_constraint(np.array([[1, 1], [2, 1]]), np.array([5, 8]), '<=')
        self.lp.solve()
        
        obj_val = self.lp.objective_value()
        self.assertIsInstance(obj_val, (int, float))
        
if __name__ == '__main__':
    unittest.main()
