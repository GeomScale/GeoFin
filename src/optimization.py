'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
### OPTIMIZATION
############################################################################

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from helper_functions import to_numpy
from covariance import Covariance
from mean_estimation import MeanEstimator
from constraints import Constraints
from optimization_data import OptimizationData
import qp_problems

# https://github.com/qpsolvers/qpsolvers


class OptimizationParameter(dict):
    """
    A class to handle parameters for optimization.

    Attributes
    ----------
    solver_name : str
        The solver to use for the optimization. Default is 'cvxopt'.
    verbose : bool
        Whether to enable verbose output. Default is True.
    allow_suboptimal : bool
        Whether to allow suboptimal solutions. Default is False.
    """

    def __init__(self, **kwargs):
        """
        Initializes the OptimizationParameter instance with default values.

        Parameters
        ----------
        **kwargs :
            Additional parameters to override defaults.
        """
        super(OptimizationParameter, self).__init__(**kwargs)
        self.__dict__ = self
        if not self.get('solver_name'):
            self['solver_name'] = 'cvxopt'
        if not self.get('verbose'):
            self['verbose'] = True
        if not self.get('allow_suboptimal'):
            self['allow_suboptimal'] = False


class Objective(dict):
    """
    A class to define optimization objectives.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the Objective instance.

        Parameters
        ----------
        *args :
            Positional arguments for the dictionary.
        **kwargs :
            Keyword arguments for the dictionary.
        """
        super(Objective, self).__init__(*args, **kwargs)


class Optimization(ABC):
    """
    Abstract base class for optimization problems.

    Attributes
    ----------
    params : OptimizationParameter
        Parameters for the optimization process.
    objective : Objective
        The optimization objective.
    constraints : Constraints
        Constraints applied to the optimization.
    model : Any
        The optimization model.
    results : dict
        Results of the optimization.
    """

    def __init__(self,
                 params: OptimizationParameter = None,
                 constraints: Constraints = None,
                 **kwargs):
        """
        Initializes the Optimization instance.

        Parameters
        ----------
        params : OptimizationParameter, optional
            Parameters for the optimization. Defaults to None.
        constraints : Constraints, optional
            Constraints for the optimization. Defaults to None.
        **kwargs :
            Additional parameters to override defaults.
        """
        self.params = OptimizationParameter(**kwargs) if params is None else params
        self.objective = Objective()
        self.constraints = Constraints() if constraints is None else constraints
        self.model = None
        self.results = None

    @abstractmethod
    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Abstract method to set the optimization objective.

        Parameters
        ----------
        optimization_data : OptimizationData
            The data used to define the objective.
        """
        raise NotImplementedError("Method 'set_objective' must be implemented in derived class.")

    @abstractmethod
    def solve(self) -> bool:
        """
        Abstract method to solve the optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        self.solve_qpsolvers()
        return self.results['status']

    def solve_qpsolvers(self) -> None:
        """
        Solves the optimization problem using qpsolvers.

        Returns
        -------
        None
        """
        self.model_qpsolvers()
        self.model.solve()
        universe = self.constraints.selection
        solution = self.model['solution']
        status = solution.found
        weights = pd.Series(solution.x[:len(universe)] if status else [None] * len(universe),
                            index=universe)

        self.results = {'weights': weights.to_dict(),
                        'status': self.model['solution'].found}

    def model_qpsolvers(self) -> None:
        """
        Sets up the quadratic programming problem for qpsolvers.

        Returns
        -------
        None
        """
        # Ensure that P and q are numpy arrays
        if 'P' in self.objective.keys():
            P = to_numpy(self.objective['P'])
        else:
            raise ValueError("Missing matrix 'P' in objective.")
        if 'q' in self.objective.keys():
            q = to_numpy(self.objective['q'])
        else:
            q = np.zeros(len(self.constraints.selection))

        self.objective['P'] = P
        self.objective['q'] = q

        universe = self.constraints.selection

        # Constraints
        constraints = self.constraints
        GhAb = constraints.to_GhAb()

        lb = constraints.box['lower'].to_numpy() if constraints.box['box_type'] != 'NA' else None
        ub = constraints.box['upper'].to_numpy() if constraints.box['box_type'] != 'NA' else None

        self.model = qp_problems.QuadraticProgram(P=self.objective['P'],
                                                  q=self.objective['q'],
                                                  constant=self.objective.get('constant'),
                                                  G=GhAb['G'],
                                                  h=GhAb['h'],
                                                  A=GhAb['A'],
                                                  b=GhAb['b'],
                                                  lb=lb,
                                                  ub=ub,
                                                  params=self.params)
        return None



class EmptyOptimization(Optimization):

    def set_objective(self) -> None:
        pass

    def solve(self) -> bool:
        return super().solve()


class MeanVariance(Optimization):
    """
    Mean-variance optimization problem.

    Attributes
    ----------
    covariance : Covariance
        Covariance estimator used in the optimization.
    mean_estimator : MeanEstimator
        Mean return estimator used in the optimization.
    """

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 mean_estimator: Optional[MeanEstimator] = None,
                 **kwargs):
        """
        Initializes the MeanVariance instance.

        Parameters
        ----------
        covariance : Covariance, optional
            Covariance estimator. Defaults to None.
        mean_estimator : MeanEstimator, optional
            Mean return estimator. Defaults to None.
        **kwargs :
            Additional parameters for the optimization.
        """
        super().__init__(**kwargs)
        self.covariance = Covariance() if covariance is None else covariance
        self.mean_estimator = MeanEstimator() if mean_estimator is None else mean_estimator
        self.params.setdefault('risk_aversion', 1)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the mean-variance optimization objective.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to compute the objective.

        Returns
        -------
        None
        """
        covmat = self.covariance.estimate(X=optimization_data['return_series'])
        covmat = covmat * self.params['risk_aversion'] * 2
        mu = self.mean_estimator.estimate(X=optimization_data['return_series']) * (-1)
        self.objective = Objective(q=mu, P=covmat)
        return None

    def solve(self) -> bool:
        """
        Solves the mean-variance optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        return super().solve()



class QEQW(Optimization):
    """
    Quasi-Equal Weighted (QEQW) optimization problem.

    Attributes
    ----------
    covariance : Covariance
        Covariance estimator used for QEQW optimization.
    """

    def __init__(self, **kwargs):
        """
        Initializes the QEQW optimization instance.

        Parameters
        ----------
        **kwargs :
            Additional parameters for the optimization.
        """
        super().__init__(**kwargs)
        self.covariance = Covariance(method='duv')

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the optimization objective for QEQW.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to define the objective.

        Returns
        -------
        None
        """
        X = optimization_data['return_series']
        covmat = self.covariance.estimate(X=X) * 2
        mu = np.zeros(X.shape[1])
        self.objective = Objective(P=covmat, q=mu)
        return None

    def solve(self) -> bool:
        """
        Solves the QEQW optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        return super().solve()


class LeastSquares(Optimization):
    """
    Least Squares optimization problem.

    Attributes
    ----------
    covariance : Covariance, optional
        Covariance estimator used in the optimization.
    """

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 **kwargs):
        """
        Initializes the Least Squares optimization instance.

        Parameters
        ----------
        covariance : Covariance, optional
            Covariance estimator. Defaults to None.
        **kwargs :
            Additional parameters for the optimization.
        """
        super().__init__(**kwargs)
        self.covariance = covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the Least Squares optimization objective.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to compute the objective.

        Returns
        -------
        None
        """
        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None and l2_penalty != 0:
            P += 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(P=P, q=q, constant=constant)
        return None

    def solve(self) -> bool:
        """
        Solves the Least Squares optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        return super().solve()


class WeightedLeastSquares(Optimization):
    """
    Weighted Least Squares optimization problem.
    """

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the Weighted Least Squares optimization objective.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to compute the objective.

        Returns
        -------
        None
        """
        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        tau = self.params['tau']
        lambda_val = np.exp(-np.log(2) / tau)
        i = np.arange(X.shape[0])
        wt_tmp = lambda_val ** i
        wt = np.flip(wt_tmp / np.sum(wt_tmp) * len(wt_tmp))
        W = np.diag(wt)

        P = 2 * ((X.T).to_numpy() @ W @ X)
        q = -2 * (X.T).to_numpy() @ W @ y
        constant = (y.T).to_numpy() @ W @ y

        self.objective = Objective(P=P,
                                   q=q,
                                   constant=constant)
        return None

    def solve(self) -> bool:
        """
        Solves the Weighted Least Squares optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """    
        return super().solve()



class LAD(Optimization):
    """
    Least Absolute Deviation (LAD) optimization problem, also known as mean absolute deviation (MAD).

    Attributes
    ----------
    params : dict
        Parameters for the optimization problem.
    objective : Objective
        Objective function for the optimization problem.
    model : QuadraticProgram
        Optimization model instance.
    results : dict
        Optimization results including weights.
    """

    def __init__(self, **kwargs):
        """
        Initializes the LAD optimization instance.

        Parameters
        ----------
        **kwargs :
            Additional parameters for the optimization.
        """
        super().__init__(**kwargs)
        self.params['use_level'] = self.params.get('use_level', True)
        self.params['use_log'] = self.params.get('use_log', True)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the LAD optimization objective.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to compute the objective.

        Returns
        -------
        None
        """
        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('use_level'):
            X = (1 + X).cumprod()
            y = (1 + y).cumprod()
            if self.params.get('use_log'):
                X = np.log(X)
                y = np.log(y)

        self.objective = Objective(X=X, y=y)
        return None

    def solve(self) -> bool:
        """
        Solves the LAD optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        self.model_qpsolvers()
        self.model.solve()
        weights = pd.Series(self.model['solution'].x[0:len(self.constraints.selection)],
                            index=self.constraints.selection)
        self.results = {'weights': weights.to_dict()}
        return True

    def model_qpsolvers(self) -> None:
        """
        Constructs the optimization model using QP solvers.

        Returns
        -------
        None
        """
        # Data and constraints
        X = to_numpy(self.objective['X'])
        y = to_numpy(self.objective['y'])
        GhAb = self.constraints.to_GhAb()
        N = X.shape[1]
        T = X.shape[0]

        # Inequality constraints
        G_tilde = np.pad(GhAb['G'], [(0, 0), (0, 2 * T)]) if GhAb['G'] is not None else None
        h_tilde = GhAb['h']

        # Equality constraints
        A = GhAb['A']
        meq = 0 if A is None else 1 if A.ndim == 1 else A.shape[0]

        A_tilde = np.zeros(shape=(T, N + 2 * T)) if A is None else np.pad(A, [(0, T), (0, 2 * T)])
        A_tilde[meq:(T + meq), 0:N] = X
        A_tilde[meq:(T + meq), N:(N + T)] = np.eye(T)
        A_tilde[meq:(T + meq), (N + T):] = -np.eye(T)

        b_tilde = y if GhAb['b'] is None else np.append(GhAb['b'], y)

        lb = to_numpy(self.constraints.box['lower']) if self.constraints.box['box_type'] != 'NA' else np.full(N, -np.inf)
        lb = np.pad(lb, (0, 2 * T))

        ub = to_numpy(self.constraints.box['upper']) if self.constraints.box['box_type'] != 'NA' else np.full(N, np.inf)
        ub = np.pad(ub, (0, 2 * T), constant_values=np.inf)

        # Objective function
        q = np.append(np.zeros(N), np.ones(2 * T))
        P = np.diag(np.zeros(N + 2 * T))

        if 'leverage' in self.constraints.l1.keys():
            lev_budget = self.constraints.l1['leverage']['rhs']
            # Auxiliary variables to deal with the abs() function
            A_tilde = np.pad(A_tilde, [(0, 0), (0, 2 * N)])
            lev_eq = np.hstack((np.eye(N), np.zeros((N, 2 * T)), -np.eye(N), np.eye(N)))
            A_tilde = np.vstack((A_tilde, lev_eq))
            b_tilde = np.append(b_tilde, np.zeros())

            G_tilde = np.pad(G_tilde, [(0, 0), (0, 2 * N)])
            lev_ineq = np.append(np.zeros(N + 2 * T), np.ones(2 * N))
            G_tilde = np.vstack((G_tilde, lev_ineq))
            h_tilde = np.append(GhAb['h'], [lev_budget])

            lb = np.pad(lb, (0, 2 * N))
            ub = np.pad(lb, (0, 2 * N), constant_values=np.inf)

        self.model = qp_problems.QuadraticProgram(P=P,
                                                  q=q,
                                                  G=G_tilde,
                                                  h=h_tilde,
                                                  A=A_tilde,
                                                  b=b_tilde,
                                                  lb=lb,
                                                  ub=ub,
                                                  params=self.params)
        return None



class PercentilePortfolios(Optimization):
    """
    Percentile-based portfolio optimization.

    Attributes
    ----------
    estimator : MeanEstimator, optional
        Estimator for computing mean returns.
    params : dict
        Parameters for the optimization problem.
    objective : Objective
        Objective function for the optimization problem.
    results : dict
        Optimization results including weights and portfolio allocations.
    """

    def __init__(self, 
                 field: Optional[str] = None,
                 estimator: Optional[MeanEstimator] = None,
                 n_percentiles: int = 5,
                 **kwargs):
        """
        Initializes the Percentile Portfolios optimization instance.

        Parameters
        ----------
        field : str, optional
            Field for scoring data.
        estimator : MeanEstimator, optional
            Estimator for mean return computation.
        n_percentiles : int, optional
            Number of percentiles. Defaults to 5 (quintile portfolios).
        **kwargs :
            Additional parameters for the optimization.
        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.params = {'solver_name': 'percentile',
                       'n_percentiles': n_percentiles,
                       'field': field}

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Sets the objective for Percentile Portfolios optimization.

        Parameters
        ----------
        optimization_data : OptimizationData
            Data used to compute the objective.

        Returns
        -------
        None
        """
        field = self.params.get('field')
        if self.estimator is not None:
            if field is not None:
                raise ValueError('Either specify a "field" or pass an "estimator", but not both.')
            else:
                scores = self.estimator.estimate(X=optimization_data['return_series'])
        else:
            if field is not None:
                scores = optimization_data['scores'][field]
            else:
                score_weights = self.params.get('score_weights')
                if score_weights is not None:
                    scores = (
                        optimization_data['scores'][score_weights.keys()]
                        .multiply(score_weights.values())
                        .sum(axis=1)
                    )
                else:
                    scores = optimization_data['scores'].mean(axis=1).squeeze()

        scores[scores == 0] = np.random.normal(0, 1e-10, scores[scores == 0].shape)
        self.objective = Objective(scores=-scores)
        return None

    def solve(self) -> bool:
        """
        Solves the Percentile Portfolios optimization problem.

        Returns
        -------
        bool
            Whether the optimization was successful.
        """
        scores = self.objective['scores']
        N = self.params['n_percentiles']
        q_vec = np.linspace(0, 100, N + 1)
        th = np.percentile(scores, q_vec)
        lID = []
        w_dict = {}
        for i in range(1, len(th)):
            if i == 1:
                lID.append(list(scores.index[scores <= th[i]]))
            else:
                lID.append(list(scores.index[np.logical_and(scores > th[i-1], scores <= th[i])]))
            w_dict[i] = scores[lID[i-1]] * 0 + 1 / len(lID[i-1])     
        weights = scores * 0
        weights[w_dict[1].keys()] = 1 / len(w_dict[1].keys())
        weights[w_dict[N].keys()] = -1 / len(w_dict[N].keys())
        self.results = {'weights': weights.to_dict(),
                        'w_dict': w_dict}
        return True
