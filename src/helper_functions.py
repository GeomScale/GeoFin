'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
### HELPER FUNCTIONS
############################################################################

from typing import Optional
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from portfolio import Portfolio, Strategy


############################################################################
# Matrix Utilities
############################################################################

def nearestPD(A: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive-definite matrix to input matrix A.

    This is a Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1],
    which credits [2]. The code below is written by Cyril.

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
         matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The nearest positive-definite matrix.
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    k = 1
    while not isPD(A3):
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B: np.ndarray) -> bool:
    """
    Returns true when input matrix B is positive-definite via Cholesky decomposition.

    Args:
        B (np.ndarray): The matrix to check.

    Returns:
        bool: True if B is positive-definite, False otherwise.
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


############################################################################
# Serialization and Data Conversion Helpers
############################################################################

def serialize_solution(name_suffix: str, solution, runtime: float) -> None:
    """
    Serialize the solution object and save it to a pickle file.

    Args:
        name_suffix (str): Suffix for the filename.
        solution: An optimization solution with attributes x, obj, and methods for residuals.
        runtime (float): The runtime of the solution.
    """
    result = {
        'solution': solution.x,
        'objective': solution.obj,
        'primal_residual': solution.primal_residual(),
        'dual_residual': solution.dual_residual(),
        'duality_gap': solution.duality_gap(),
        'runtime': runtime
    }

    with open(f'{name_suffix}.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def to_numpy(data):
    """
    Convert data to a NumPy array if possible.

    Args:
        data: Input data that might have a 'to_numpy' method.

    Returns:
        The NumPy array or None if data is None.
    """
    return None if data is None else data.to_numpy() if hasattr(data, 'to_numpy') else data


############################################################################
# Portfolio Strategy Conversion
############################################################################

def output_to_strategies(output: dict) -> dict:
    """
    Convert output dictionary into a dictionary of Strategy instances.

    Each key in the output corresponds to a rebalance date and each strategy is 
    constructed from the weights provided.

    Args:
        output (dict): Dictionary with keys as rebalance dates and values as dictionaries 
                       containing weights for each strategy.

    Returns:
        dict: A dictionary mapping strategy keys (e.g., 'q1', 'q2', ...) to Strategy objects.
    """
    N = len(output[list(output.keys())[0]])
    strategy_dict = {}
    for i in range(N):
        key = f'q{i+1}'
        strategy_dict[key] = Strategy([])
        for rebdate in output.keys():
            weights = output[rebdate][f'weights_{i+1}']
            if hasattr(weights, 'to_dict'):
                weights = weights.to_dict()
            portfolio = Portfolio(rebdate, weights)
            strategy_dict[key].portfolios.append(portfolio)

    return strategy_dict


############################################################################
# Machine Learning Helpers
############################################################################

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values (can be a pandas object with a 'values' attribute).

    Returns:
        float: The RMSE value.
    """
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred.values)) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) in percent.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        float: The MAPE value.
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def show_result(predictions, y_test, y_actual, method=None):
    """
    Display RMSE and MAPE metrics and plot predictions versus true values.

    Args:
        predictions: Predicted values.
        y_test: Values used to calculate metrics.
        y_actual: Actual values for plotting.
        method (optional): Method or model name for display purposes.
    """
    print(f'RMSE of linear regression: {calculate_rmse(y_test, predictions)}')
    print(f'MAPE of linear regression: {calculate_mape(y_test, predictions)}')

    plt.plot(y_actual, color='cyan')
    plt.plot(predictions, color='green')
    plt.legend(["True values", "Prediction"])
    plt.title(method)
    plt.show()
