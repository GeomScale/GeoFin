'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

import pandas as pd
import numpy as np

from helper_functions import isPD, nearestPD


class CovarianceSpecification(dict):
    """
    Configuration class for specifying covariance estimation parameters.

    Attributes
    ----------
    method : str
        The method to estimate the covariance matrix. Default is 'pearson'.
    check_positive_definite : bool
        Whether to ensure the covariance matrix is positive definite. Default is True.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a CovarianceSpecification instance with default values if not provided.

        Parameters
        ----------
        *args :
            Positional arguments passed to the dictionary initialization.
        **kwargs :
            Keyword arguments for covariance specification.
        """
        super(CovarianceSpecification, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # Add default values
        if self.get('method') is None:
            self['method'] = 'pearson'
        if self.get('check_positive_definite') is None:
            self['check_positive_definite'] = True



class Covariance:
    """
    Class to estimate covariance matrices based on a specified configuration.

    Attributes
    ----------
    spec : CovarianceSpecification
        Configuration specifying the estimation parameters and settings.
    """

    def __init__(self, spec: CovarianceSpecification = None, *args, **kwargs):
        """
        Initializes a Covariance instance with a given specification or default values.

        Parameters
        ----------
        spec : CovarianceSpecification, optional
            A pre-defined specification for covariance estimation. If None, uses default settings.
        *args :
            Additional positional arguments to define the specification.
        **kwargs :
            Additional keyword arguments to define the specification.
        """
        self.spec = CovarianceSpecification(*args, **kwargs) if spec is None else spec



    def set_ctrl(self, *args, **kwargs) -> None:
        """
        Updates the covariance estimation specification.

        Parameters
        ----------
        *args :
            Positional arguments for updating the specification.
        **kwargs :
            Keyword arguments for updating the specification.
        """
        self.spec = CovarianceSpecification(*args, **kwargs)



    def estimate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates the covariance matrix based on the provided data and specified method.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for which the covariance matrix is to be estimated.

        Returns
        -------
        pd.DataFrame
            The estimated covariance matrix.

        Raises
        ------
        NotImplementedError
            If the specified method is not implemented.
        """
        estimation_method = self.spec['method']
        if estimation_method == 'pearson':
            covmat = cov_pearson(X)
        elif estimation_method == 'duv':
            covmat = cov_duv(X)
        elif estimation_method == 'linear_shrinkage':
            lambda_covmat_regularization = self.spec.get('lambda_covmat_regularization')
            covmat = cov_linear_shrinkage(X, lambda_covmat_regularization)
        else:
            raise NotImplementedError('This method is not implemented yet')
        if self.spec.get('check_positive_definite'):
            if not isPD(covmat):
                covmat = nearestPD(covmat)

        return covmat


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def cov_pearson(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pearson covariance matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Pearson covariance matrix.
    """
    return X.cov()


def cov_duv(X: pd.DataFrame) -> np.ndarray:
    """
    Returns a diagonal unit variance covariance matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    Returns
    -------
    np.ndarray
        Diagonal covariance matrix with ones on the diagonal.
    """
    return np.identity(X.shape[1])


def cov_linear_shrinkage(X: pd.DataFrame, lambda_covmat_regularization: float = None) -> pd.DataFrame:
    """
    Applies linear shrinkage to the covariance matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.
    lambda_covmat_regularization : float, optional
        Regularization parameter. Default is None, which assumes no shrinkage.

    Returns
    -------
    pd.DataFrame
        Regularized covariance matrix.
    """
    if lambda_covmat_regularization is None or np.isnan(lambda_covmat_regularization) or lambda_covmat_regularization < 0:
        lambda_covmat_regularization = 0
    sigmat = X.cov()
    if lambda_covmat_regularization > 0:
        d = sigmat.shape[0]
        sig = np.sqrt(np.diag(sigmat.to_numpy()))
        corrMat = np.diag(1.0 / sig) @ sigmat.to_numpy() @ np.diag(1.0 / sig)
        corrs = []
        for k in range(1, d):
            corrs.extend(np.diag(corrMat, k))
        sigmat = pd.DataFrame(sigmat.to_numpy() + lambda_covmat_regularization * np.mean(sig**2) * np.eye(d),
                              columns=sigmat.columns, index=sigmat.index)
    return sigmat

