'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

import numpy as np
import qpsolvers
import scipy
import pickle
from helper_functions import isPD, nearestPD

IGNORED_SOLVERS = {'gurobi', 'mosek', 'ecos', 'scs', 'piqp', 'proxqp', 'clarabel'}
SPARSE_SOLVERS = {'clarabel', 'ecos', 'gurobi', 'mosek', 'highs', 'qpalm', 'osqp', 'qpswift', 'scs'}
ALL_SOLVERS = {'clarabel', 'cvxopt', 'daqp', 'ecos', 'gurobi', 'highs', 'mosek', 'osqp', 'piqp', 'proxqp', 'qpalm', 'quadprog', 'scs'}
USABLE_SOLVERS = ALL_SOLVERS - IGNORED_SOLVERS

class QuadraticProgram(dict):
    """
    A class for representing and solving quadratic optimization problems.

    This class provides methods for transforming financial optimization problems into
    a standard quadratic programming format.

    Attributes
    ----------
    solver : str
        The name of the solver used for optimization.

    Methods
    -------
    linearize_turnover_constraint(x_init, to_budget)
        Adds a turnover constraint by introducing auxiliary variables.
    linearize_leverage_constraint(N, leverage_budget)
        Adds a leverage constraint by introducing auxiliary variables.
    linearize_turnover_objective(x_init, transaction_cost)
        Modifies the objective function to account for transaction costs.
    is_feasible()
        Checks if the quadratic program is feasible.
    solve()
        Solves the quadratic optimization problem.
    objective_value(x, with_const=True)
        Computes the objective function value for a given solution.
    serialize(path, **kwargs)
        Saves the quadratic program to a file.
    load(path, **kwargs)
        Loads a quadratic program from a file.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the QuadraticProgram instance.

        Parameters
        ----------
        *args :
            Positional arguments passed to the dictionary constructor.
        **kwargs :
            Keyword arguments passed to the dictionary constructor.
        """
        super(QuadraticProgram, self).__init__(*args, **kwargs)
        self.solver = self['params']['solver_name']


    def linearize_turnover_constraint(self, x_init: np.ndarray, to_budget: float = float('inf')) -> None:
        """
        Adds turnover constraints to the quadratic program by introducing auxiliary variables.

        Parameters
        ----------
        x_init : np.ndarray
            Initial portfolio weights before rebalancing.
        to_budget : float, optional
            Maximum allowed turnover budget, by default infinity.

        Returns
        -------
        None

        Notes
        -----
        - This method modifies the objective function and constraints to account for turnover.
        - It extends the quadratic program with additional variables to track turnover.
        """
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Extend matrices for turnover constraints
        P = np.pad(self['P'], (0, n)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, n)) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(m + 2 * n + 1, 2 * n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m + n), 0:n] = np.eye(n)
        G[m:(m + n), n:(2 * n)] = -np.eye(n)
        G[(m + n):(m + 2 * n), 0:n] = -np.eye(n)
        G[(m + n):(m + 2 * n), n:(2 * n)] = -np.eye(n)
        G[(m + 2 * n),] = np.append(np.zeros(n), np.ones(n))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(np.append(x_init, -x_init), to_budget))

        # Equality constraints
        A = np.pad(self['A'], [(0, 0), (0, n)]) if self.get('A') is not None else None

        # Adjust bounds
        lb = np.pad(self['lb'], (0, n)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, n), constant_values=float('inf')) if self.get('ub') is not None else None

        # Update problem
        self.update({'P': P, 'q': q, 'G': G, 'h': h, 'A': A, 'lb': lb, 'ub': ub})

        return None

    def linearize_leverage_constraint(self, N: int = None, leverage_budget: float = 2) -> None:
        """
        Adds leverage constraints to the quadratic program by introducing auxiliary variables.

        Parameters
        ----------
        N : int
            Number of assets.
        leverage_budget : float, optional
            Maximum leverage allowed, by default 2.

        Returns
        -------
        None

        Notes
        -----
        - This method extends the quadratic program to account for leverage constraints.
        - It introduces auxiliary variables to ensure that leverage does not exceed the defined budget.
        """
        # Dimensions
        n = len(self.get('q'))
        mG = 0 if self.get('G') is None else self.get('G').shape[0]
        mA = 1 if self.get('A').ndim == 1 else self.get('A').shape[0]

        # Extend matrices for leverage constraints
        P = np.pad(self['P'], (0, 2 * N)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, 2 * N)) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(mG + 1, n + 2 * N))
        if self.get('G') is not None:
            G[0:mG, 0:n] = self.get('G')
        G[mG,] = np.append(np.zeros(n), np.ones(2 * N))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, leverage_budget)

        # Equality constraints
        A = np.zeros(shape=(mA + N, n + 2 * N))
        A[0:mA, 0:n] = self.get('A')
        A[mA:(mA + N), 0:N] = np.eye(N)
        A[mA:(mA + N), n:(n + N)] = np.eye(N)
        A[mA:(mA + N), (n + N):(n + 2 * N)] = -np.eye(N)
        b = np.pad(self.get('b'), (0, N))

        # Adjust bounds
        lb = np.pad(self['lb'], (0, 2 * N)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, 2 * N), constant_values=float('inf')) if self.get('ub') is not None else None

        # Update problem
        self.update({'P': P, 'q': q, 'G': G, 'h': h, 'A': A, 'b': b, 'lb': lb, 'ub': ub})

        return None

    def linearize_turnover_objective(self, x_init: np.ndarray, transaction_cost: float = 0.002) -> None:
        """
        Modifies the objective function to include transaction costs in the turnover.

        Parameters
        ----------
        x_init : np.ndarray
            Initial portfolio weights before rebalancing.
        transaction_cost : float, optional
            Cost per unit turnover, by default 0.002.

        Returns
        -------
        None

        Notes
        -----
        - This method introduces additional variables to the optimization problem
          to model turnover-related transaction costs.
        - The objective function is modified to penalize excessive turnover.
        """
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Extend matrices for turnover objective
        P = np.pad(self['P'], (0, n)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, n), constant_values=transaction_cost) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(m + 2 * n, 2 * n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m + n), 0:n] = np.eye(n)
        G[m:(m + n), n:(2 * n)] = -np.eye(n)
        G[(m + n):(m + 2 * n), 0:n] = -np.eye(n)
        G[(m + n):(m + 2 * n), n:(2 * n)] = -np.eye(n)
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(x_init, -x_init))

        # Equality constraints
        A = np.pad(self['A'], [(0, 0), (0, n)]) if self.get('A') is not None else None

        # Adjust bounds
        lb = np.pad(self['lb'], (0, n)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, n), constant_values=float('inf')) if self.get('ub') is not None else None

        # Update problem
        self.update({'P': P, 'q': q, 'G': G, 'h': h, 'A': A, 'lb': lb, 'ub': ub})

        return None


    def is_feasible(self) -> bool:
        """
        Checks whether the quadratic program is feasible.

        Returns
        -------
        bool
            True if the program is feasible, otherwise False.
        """
        problem = qpsolvers.Problem(P=np.zeros(self.get('P').shape),
                                    q=np.zeros(self.get('P').shape[0]),
                                    G=self.get('G'),
                                    h=self.get('h'),
                                    A=self.get('A'),
                                    b=self.get('b'),
                                    lb=self.get('lb'),
                                    ub=self.get('ub'))

        if self.solver in SPARSE_SOLVERS and self['params'].get('sparse'):
            problem.P = scipy.sparse.csc_matrix(problem.P) if problem.P is not None else None
            problem.A = scipy.sparse.csc_matrix(problem.A) if problem.A is not None else None
            problem.G = scipy.sparse.csc_matrix(problem.G) if problem.G is not None else None

        solution = qpsolvers.solve_problem(problem=problem, solver=self.solver, initvals=self.get('x0'), verbose=False)
        return solution.found

    def solve(self) -> None:
        """
        Solves the quadratic optimization problem.

        Returns
        -------
        None
        """
        if self.solver in ['ecos', 'scs', 'clarabel']:
            if self.get('b').size == 1:
                self['b'] = np.array(self.get('b')).reshape(-1)

        P = self.get('P')
        if P is not None and not isPD(P):
            self['P'] = nearestPD(P)

        problem = qpsolvers.Problem(P=self.get('P'),
                                    q=self.get('q'),
                                    G=self.get('G'),
                                    h=self.get('h'),
                                    A=self.get('A'),
                                    b=self.get('b'),
                                    lb=self.get('lb'),
                                    ub=self.get('ub'))

        if self.solver in SPARSE_SOLVERS and self['params'].get('sparse'):
            problem.P = scipy.sparse.csc_matrix(problem.P) if problem.P is not None else None
            problem.A = scipy.sparse.csc_matrix(problem.A) if problem.A is not None else None
            problem.G = scipy.sparse.csc_matrix(problem.G) if problem.G is not None else None

        solution = qpsolvers.solve_problem(problem=problem, solver=self.solver, initvals=self.get('x0'), verbose=False)
        self['solution'] = solution
        return None

    def objective_value(self, x: np.ndarray, with_const: bool = True) -> float:
        """
        Computes the objective function value for a given solution.

        Parameters
        ----------
        x : np.ndarray
            Decision variable values.
        with_const : bool, optional
            Whether to include the constant term, by default True.

        Returns
        -------
        float
            The computed objective function value.
        """
        const = 0 if self.get('constant') is None or not with_const else self['constant']
        return (0.5 * (x @ self.get('P') @ x) + self.get('q') @ x).item() + const

    def serialize(self, path: str, **kwargs) -> None:
        """
        Saves the QuadraticProgram instance to a file.

        Parameters
        ----------
        path : str
            Path to save the serialized file.
        **kwargs :
            Additional arguments for pickle.

        Returns
        -------
        None
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f, kwargs)

    @staticmethod
    def load(path: str, **kwargs) -> 'QuadraticProgram':
        with open(path, 'rb') as f:
            return pickle.load(f, **kwargs)
