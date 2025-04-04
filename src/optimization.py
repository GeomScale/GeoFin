'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
# OPTIMIZATION MODULE
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
import qp_problems  # https://github.com/qpsolvers/qpsolvers


class OptimizationParameter(dict):
    """
    Dictionary-like container for optimization parameters with sensible defaults.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self
        self.setdefault('solver_name', 'cvxopt')
        self.setdefault('verbose', True)
        self.setdefault('allow_suboptimal', False)


class Objective(dict):
    """
    Container for the optimization objective.
    """
    pass


class Optimization(ABC):
    """
    Abstract base class for optimization models.
    """

    def __init__(self,
                 params: Optional[OptimizationParameter] = None,
                 constraints: Optional[Constraints] = None,
                 **kwargs):
        self.params = params if params is not None else OptimizationParameter(**kwargs)
        self.objective = Objective()
        self.constraints = constraints if constraints is not None else Constraints()
        self.model = None
        self.results = None

    @abstractmethod
    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Construct the objective based on the optimization data.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("Method 'set_objective' must be implemented in derived class.")

    def solve(self) -> bool:
        """
        Solve the optimization problem using the qpsolvers backend.
        Returns a boolean indicating if the solution was found.
        """
        self.solve_qpsolvers()
        return self.results.get('status', False)

    def solve_qpsolvers(self) -> None:
        """
        Build and solve the quadratic programming problem.
        """
        self.model_qpsolvers()
        self.model.solve()
        universe = self.constraints.selection
        solution = self.model['solution']
        status = solution.found
        weights = pd.Series(
            solution.x[:len(universe)] if status else [None] * len(universe),
            index=universe
        )
        self.results = {'weights': weights.to_dict(),
                        'status': status}

    def model_qpsolvers(self) -> None:
        """
        Prepare the QP model using the objective and constraints.
        """
        # Ensure that P is provided; default q to zeros if missing
        if 'P' in self.objective:
            P = to_numpy(self.objective['P'])
        else:
            raise ValueError("Missing matrix 'P' in objective.")

        q = to_numpy(self.objective.get('q', np.zeros(len(self.constraints.selection))))
        self.objective['P'] = P
        self.objective['q'] = q

        universe = self.constraints.selection
        ghab = self.constraints.to_GhAb()

        lb = (self.constraints.box['lower'].to_numpy() 
              if self.constraints.box['box_type'] != 'NA' else None)
        ub = (self.constraints.box['upper'].to_numpy() 
              if self.constraints.box['box_type'] != 'NA' else None)

        self.model = qp_problems.QuadraticProgram(
            P=self.objective['P'],
            q=self.objective['q'],
            constant=self.objective.get('constant'),
            G=ghab['G'],
            h=ghab['h'],
            A=ghab['A'],
            b=ghab['b'],
            lb=lb,
            ub=ub,
            params=self.params
        )

        # Set reference position for turnover constraints if provided
        turnover = self.constraints.l1.get('turnover')
        x0 = (turnover.get('x0') if turnover is not None and turnover.get('x0') is not None
              else self.params.get('x0'))
        x_init = {asset: x0.get(asset, 0) for asset in universe} if x0 is not None else None

        transaction_cost = self.params.get('transaction_cost')
        if transaction_cost is not None and x_init is not None:
            self.model.linearize_turnover_objective(pd.Series(x_init), transaction_cost)
        elif turnover and not transaction_cost and x_init is not None:
            self.model.linearize_turnover_constraint(pd.Series(x_init), turnover['rhs'])

        # Leverage constraint linearization
        levcon = self.constraints.l1.get('leverage')
        if levcon is not None:
            self.model.linearize_leverage_constraint(N=len(universe), leverage_budget=levcon['rhs'])


class EmptyOptimization(Optimization):
    """
    Dummy optimization implementation.
    """

    def set_objective(self, optimization_data: OptimizationData) -> None:
        pass

    def solve(self) -> bool:
        return super().solve()


class MeanVariance(Optimization):
    """
    Mean-Variance optimization implementation.
    """

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 mean_estimator: Optional[MeanEstimator] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.covariance = covariance if covariance is not None else Covariance()
        # Fix: use provided mean_estimator or instantiate one
        self.mean_estimator = mean_estimator if mean_estimator is not None else MeanEstimator()
        self.params.setdefault('risk_aversion', 1)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        returns = optimization_data['return_series']
        covmat = self.covariance.estimate(X=returns)
        # Scale covariance matrix by risk aversion (multiplied by 2 for QP formulation)
        covmat *= self.params['risk_aversion'] * 2
        mu = self.mean_estimator.estimate(X=returns) * (-1)
        self.objective = Objective(q=mu, P=covmat)

    def solve(self) -> bool:
        return super().solve()


class QEQW(Optimization):
    """
    Equal-Weighted optimization with a covariance-based objective.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.covariance = Covariance(method='duv')

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        covmat = self.covariance.estimate(X=X) * 2
        mu = np.zeros(X.shape[1])
        self.objective = Objective(P=covmat, q=mu)

    def solve(self) -> bool:
        return super().solve()


class LeastSquares(Optimization):
    """
    Least squares optimization.
    """

    def __init__(self, covariance: Optional[Covariance] = None, **kwargs):
        super().__init__(**kwargs)
        self.covariance = covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        y = optimization_data['bm_series']

        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        # 0.5 * w' P w - q' w + constant
        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        # Add L2 penalty if provided
        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty:
            P += 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(P=P, q=q, constant=constant)

    def solve(self) -> bool:
        return super().solve()


class WeightedLeastSquares(Optimization):
    """
    Weighted least squares optimization.
    """

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        y = optimization_data['bm_series']

        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        tau = self.params['tau']
        lambda_val = np.exp(-np.log(2) / tau)
        indices = np.arange(X.shape[0])
        weights_tmp = lambda_val ** indices
        # Flip weights so that more recent observations get higher weight
        weights_norm = np.flip(weights_tmp / np.sum(weights_tmp) * len(weights_tmp))
        W = np.diag(weights_norm)

        # Convert DataFrame columns to numpy arrays explicitly
        P = 2 * (X.T.to_numpy() @ W @ X)
        q = -2 * (X.T.to_numpy() @ W @ y)
        constant = (y.T.to_numpy() @ W @ y)

        self.objective = Objective(P=P, q=q, constant=constant)

    def solve(self) -> bool:
        return super().solve()


class LAD(Optimization):
    """
    Least Absolute Deviation (LAD) optimization, also known as Mean Absolute Deviation (MAD).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params['use_level'] = self.params.get('use_level', True)
        self.params['use_log'] = self.params.get('use_log', True)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        y = optimization_data['bm_series']

        if self.params.get('use_level'):
            X = (1 + X).cumprod()
            y = (1 + y).cumprod()
            if self.params.get('use_log'):
                X = np.log(X)
                y = np.log(y)

        self.objective = Objective(X=X, y=y)

    def solve(self) -> bool:
        # Note: In practice an interior point linear solver might be preferred.
        self.model_qpsolvers()
        self.model.solve()
        weights = pd.Series(self.model['solution'].x[:len(self.constraints.selection)],
                            index=self.constraints.selection)
        self.results = {'weights': weights.to_dict()}
        return True

    def model_qpsolvers(self) -> None:
        """
        Build the QP model for the LAD problem.
        """
        X = to_numpy(self.objective['X'])
        y = to_numpy(self.objective['y'])
        ghab = self.constraints.to_GhAb()
        N = X.shape[1]
        T = X.shape[0]

        # Inequality constraints
        G_tilde = np.pad(ghab['G'], [(0, 0), (0, 2 * T)]) if ghab['G'] is not None else None
        h_tilde = ghab['h']

        # Equality constraints
        A = ghab['A']
        meq = 0 if A is None else (1 if A.ndim == 1 else A.shape[0])
        A_tilde = np.zeros((T, N + 2 * T)) if A is None else np.pad(A, [(0, T), (0, 2 * T)])
        A_tilde[meq:(T + meq), :N] = X
        A_tilde[meq:(T + meq), N:N + T] = np.eye(T)
        A_tilde[meq:(T + meq), N + T:] = -np.eye(T)

        b_tilde = y if ghab['b'] is None else np.append(ghab['b'], y)

        lb = to_numpy(self.constraints.box['lower']) if self.constraints.box['box_type'] != 'NA' else np.full(N, -np.inf)
        lb = np.pad(lb, (0, 2 * T))

        ub = to_numpy(self.constraints.box['upper']) if self.constraints.box['box_type'] != 'NA' else np.full(N, np.inf)
        ub = np.pad(ub, (0, 2 * T), constant_values=np.inf)

        # Objective function for LAD
        q = np.append(np.zeros(N), np.ones(2 * T))
        P = np.diag(np.zeros(N + 2 * T))

        # Leverage constraints handling
        if 'leverage' in self.constraints.l1:
            lev_budget = self.constraints.l1['leverage']['rhs']
            A_tilde = np.pad(A_tilde, [(0, 0), (0, 2 * N)])
            lev_eq = np.hstack((np.eye(N), np.zeros((N, 2 * T)), -np.eye(N), np.eye(N)))
            A_tilde = np.vstack((A_tilde, lev_eq))
            b_tilde = np.append(b_tilde, np.zeros(N))
            G_tilde = np.pad(G_tilde, [(0, 0), (0, 2 * N)])
            lev_ineq = np.append(np.zeros(N + 2 * T), np.ones(2 * N))
            G_tilde = np.vstack((G_tilde, lev_ineq))
            h_tilde = np.append(ghab['h'], [lev_budget])
            lb = np.pad(lb, (0, 2 * N))
            ub = np.pad(lb, (0, 2 * N), constant_values=np.inf)

        self.model = qp_problems.QuadraticProgram(
            P=P,
            q=q,
            G=G_tilde,
            h=h_tilde,
            A=A_tilde,
            b=b_tilde,
            lb=lb,
            ub=ub,
            params=self.params
        )


class PercentilePortfolios(Optimization):
    """
    Constructs portfolios based on percentiles of score distributions.
    """

    def __init__(self, 
                 field: Optional[str] = None,
                 estimator: Optional[MeanEstimator] = None,
                 n_percentiles: int = 5,  # Defaults to quintile portfolios.
                 **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        # Overwrite parameters with percentile-specific values
        self.params = OptimizationParameter(solver_name='percentile',
                                              n_percentiles=n_percentiles,
                                              field=field)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        """
        Define the objective based on score percentiles.
        """
        field = self.params.get('field')
        if self.estimator is not None:
            if field is not None:
                raise ValueError('Specify either a "field" or pass an "estimator", but not both.')
            scores = self.estimator.estimate(X=optimization_data['return_series'])
        else:
            if field is not None:
                scores = optimization_data['scores'][field]
            else:
                score_weights = self.params.get('score_weights')
                if score_weights is not None:
                    scores = (
                        optimization_data['scores'][list(score_weights.keys())]
                        .multiply(list(score_weights.values()))
                        .sum(axis=1)
                    )
                else:
                    scores = optimization_data['scores'].mean(axis=1).squeeze()

        # Avoid duplicated thresholds at zero by adding minimal noise
        scores[scores == 0] = np.random.normal(0, 1e-10, scores[scores == 0].shape)
        self.objective = Objective(scores=-scores)

    def solve(self) -> bool:
        """
        Constructs percentile-based portfolios and assigns long/short weights.
        """
        scores = self.objective['scores']
        n_percentiles = self.params['n_percentiles']
        percentile_edges = np.linspace(0, 100, n_percentiles + 1)
        thresholds = np.percentile(scores, percentile_edges)
        portfolio_ids = {}
        for i in range(1, len(thresholds)):
            if i == 1:
                indices = scores.index[scores <= thresholds[i]]
            else:
                indices = scores.index[(scores > thresholds[i - 1]) & (scores <= thresholds[i])]
            portfolio_ids[i] = indices
        # Build weights: assign long to the bottom percentile and short to the top percentile
        weights = scores * 0.0
        if portfolio_ids.get(1):
            long_weight = 1 / len(portfolio_ids[1])
            weights.loc[portfolio_ids[1]] = long_weight
        if portfolio_ids.get(n_percentiles):
            short_weight = -1 / len(portfolio_ids[n_percentiles])
            weights.loc[portfolio_ids[n_percentiles]] = short_weight
        self.results = {'weights': weights.to_dict(),
                        'portfolio_ids': portfolio_ids}
        return True
