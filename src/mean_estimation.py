
'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
### MEAN ESTIMATOR
############################################################################

import pandas as pd
import numpy as np

class MeanEstimator:
    """
    Estimates the expected return of financial assets using various methods.

    Attributes
    ----------
    spec : dict
        Specification for the estimation method, including:
        - 'method': The method of estimation (default: 'geometric').
        - 'scalefactor': Scaling factor for the estimated return.
        - 'n_mom': Number of moments used for estimation.
        - 'n_rev': Number of moments to reverse.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the MeanEstimator with default or user-provided specifications.

        Parameters
        ----------
        **kwargs :
            Keyword arguments to customize the specification, such as 'method', 'scalefactor',
            'n_mom', and 'n_rev'.
        """
        self.spec = {
            'method': 'geometric',
            'scalefactor': 1,
            'n_mom': None,
            'n_rev': None
        }
        self.spec.update(kwargs)

    def estimate(self, X: pd.DataFrame) -> pd.DataFrame or pd.Series:
        """
        Estimates the mean return using the specified method.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing historical returns.

        Returns
        -------
        pd.DataFrame or pd.Series
            The estimated mean return.

        Raises
        ------
        AttributeError
            If the specified method is not implemented.
        """
        fun = getattr(self, f'estimate_{self.spec["method"]}')
        mu = fun(X=X)
        return mu

    def estimate_geometric(self, X: pd.DataFrame) -> pd.Series:
        """
        Estimates the mean return using the geometric mean method.

        Parameters
        ----------
        X : pd.DataFrame
            Input data containing historical returns.

        Returns
        -------
        pd.Series
            The estimated mean return.
        """
        n_mom = X.shape[0] if self.spec.get('n_mom') is None else self.spec.get('n_mom')
        n_rev = 0 if self.spec.get('n_rev') is None else self.spec.get('n_rev')
        scalefactor = 1 if self.spec.get('scalefactor') is None else self.spec.get('scalefactor')
        X = X.tail(n_mom).head(n_mom - n_rev)
        mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1
        return mu
