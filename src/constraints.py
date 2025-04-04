'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

import warnings
import pandas as pd
import numpy as np
from typing import Dict

###############################################################################
# Constraints Class
###############################################################################

class Constraints:

    def __init__(self, selection="NA") -> None:
        """
        Initialize a Constraints instance.

        Args:
            selection (str or iterable of str): A character vector for asset selection.
                Each element must be a string.
        Raises:
            ValueError: If any element in selection is not a string.
        """
        if not all(isinstance(item, str) for item in selection):
            raise ValueError("argument 'selection' has to be a character vector.")

        self.selection = selection
        self.budget = {'Amat': None, 'sense': None, 'rhs': None}
        self.box = {'box_type': 'NA', 'lower': None, 'upper': None}
        self.linear = {'Amat': None, 'sense': None, 'rhs': None}
        self.l1 = {}

    def __str__(self) -> str:
        """
        Return a string representation of the Constraints instance,
        listing all attributes and their current values.
        """
        return ' '.join(f'\n{key}:\n\n{vars(self)[key]}\n' for key in vars(self).keys())

    def add_budget(self, rhs=1, sense='=') -> None:
        """
        Add or update the budget constraint.

        Args:
            rhs: Right-hand side value of the budget constraint.
            sense (str): Constraint sense (e.g., '='). Defaults to '='.
        """
        if self.budget.get('rhs') is not None:
            warnings.warn("Existing budget constraint is overwritten\n")

        a_values = pd.Series(np.ones(len(self.selection)), index=self.selection)
        self.budget = {'Amat': a_values,
                       'sense': sense,
                       'rhs': rhs}

    def add_box(self, box_type="LongOnly", lower=None, upper=None) -> None:
        """
        Add or update the box constraint.

        Args:
            box_type (str): The type of box constraint. Options are "LongOnly", "LongShort", or "Unbounded".
            lower: The lower bound(s).
            upper: The upper bound(s).

        Raises:
            ValueError: If any lower bound is higher than the corresponding upper bound.
        """
        boxcon = box_constraint(box_type, lower, upper)

        if np.isscalar(boxcon['lower']):
            boxcon['lower'] = pd.Series(np.repeat(float(boxcon['lower']), len(self.selection)), index=self.selection)
        if np.isscalar(boxcon['upper']):
            boxcon['upper'] = pd.Series(np.repeat(float(boxcon['upper']), len(self.selection)), index=self.selection)

        if (boxcon['upper'] < boxcon['lower']).any():
            raise ValueError("Some lower bounds are higher than the corresponding upper bounds.")

        self.box = boxcon

    def add_linear(self,
                   Amat: pd.DataFrame = None,
                   a_values: pd.Series = None,
                   sense: str = '=',
                   rhs=None,
                   name: str = None) -> None:
        """
        Add or update linear constraints.

        Args:
            Amat (pd.DataFrame, optional): Constraint matrix.
            a_values (pd.Series, optional): Alternative to Amat. If provided, used to construct Amat.
            sense (str): Constraint sense. Defaults to '='.
            rhs: Right-hand side value(s) for the constraint.
            name (str, optional): Name for the constraint (used as index in Amat).

        Raises:
            ValueError: If neither Amat nor a_values is provided.
        """
        if Amat is None:
            if a_values is None:
                raise ValueError("Either 'Amat' or 'a_values' must be provided.")
            else:
                Amat = pd.DataFrame(a_values).T.reindex(columns=self.selection).fillna(0)
                if name is not None:
                    Amat.index = [name]

        if isinstance(sense, str):
            sense = pd.Series([sense])

        if isinstance(rhs, (int, float)):
            rhs = pd.Series([rhs])

        if self.linear['Amat'] is not None:
            Amat = pd.concat([self.linear['Amat'], Amat], axis=0, ignore_index=False)
            sense = pd.concat([self.linear['sense'], sense], axis=0, ignore_index=False)
            rhs = pd.concat([self.linear['rhs'], rhs], axis=0, ignore_index=False)

        Amat.fillna(0, inplace=True)

        self.linear = {'Amat': Amat, 'sense': sense, 'rhs': rhs}

    def add_l1(self, name: str, rhs=None, x0=None, *args, **kwargs) -> None:
        """
        Add or update an l1-type constraint (e.g., for turnover or leverage).

        Args:
            name (str): Name of the constraint.
            rhs: Right-hand side value.
            x0 (optional): An initial guess.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            TypeError: If rhs is not provided.
        """
        if rhs is None:
            raise TypeError("argument 'rhs' is required.")
        con = {'rhs': rhs}
        if x0 is not None:
            con['x0'] = x0
        for i, arg in enumerate(args):
            con[f'arg{i}'] = arg
        for key, value in kwargs.items():
            con[key] = value
        self.l1[name] = con

    def to_GhAb(self, lbub_to_G: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Convert constraints into matrices/vectors for optimization.

        Args:
            lbub_to_G (bool): If True, convert lower and upper bounds from the box constraint into G and h matrices.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'G', 'h', 'A', and 'b' corresponding to constraint matrices/vectors.
        """
        A = None
        b = None
        G = None
        h = None

        if self.budget['Amat'] is not None:
            if self.budget['sense'] == '=':
                A = np.array(self.budget['Amat'], dtype=float)
                b = np.array(self.budget['rhs'], dtype=float)
            else:
                G = np.array(self.budget['Amat'], dtype=float)
                h = np.array(self.budget['rhs'], dtype=float)

        if lbub_to_G:
            I = np.eye(len(self.selection))
            G_tmp = np.concatenate((-I, I), axis=0)
            # Explicitly convert to NumPy arrays before concatenation
            lower_arr = np.array(self.box["lower"])
            upper_arr = np.array(self.box["upper"])
            h_tmp = np.concatenate((-lower_arr, upper_arr), axis=0)
            G = np.vstack((G, G_tmp)) if (G is not None) else G_tmp
            h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        if self.linear['Amat'] is not None:
            Amat = self.linear['Amat'].copy()
            rhs = self.linear['rhs'].copy()

            # Ensure that the system of inequalities is all '<='
            idx_geq = np.array(self.linear['sense'] == '>=')
            if idx_geq.sum() > 0:
                Amat[idx_geq] = -Amat[idx_geq]
                rhs[idx_geq] = -rhs[idx_geq]

            # Extract equality constraints
            idx_eq = np.array(self.linear['sense'] == '=')
            if idx_eq.sum() > 0:
                A_tmp = Amat[idx_eq].to_numpy()
                b_tmp = rhs[idx_eq].to_numpy()
                A = np.vstack((A, A_tmp)) if A is not None else A_tmp
                b = np.concatenate((b, b_tmp), axis=None) if b is not None else b_tmp
                if idx_eq.sum() < Amat.shape[0]:
                    G_tmp = Amat[np.logical_not(idx_eq)].to_numpy()
                    h_tmp = rhs[np.logical_not(idx_eq)].to_numpy()
            else:
                G_tmp = Amat.to_numpy()
                h_tmp = rhs.to_numpy()

            if 'G_tmp' in locals():
                G = np.vstack((G, G_tmp)) if G is not None else G_tmp
                h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        # To ensure A and G are matrices (even if only one row)
        A = A.reshape(-1, A.shape[-1]) if A is not None else None
        G = G.reshape(-1, G.shape[-1]) if G is not None else None

        return {'G': G, 'h': h, 'A': A, 'b': b}

###############################################################################
# Helper Functions
###############################################################################

def match_arg(x, lst):
    """
    Return the first element from lst that contains x.

    Args:
        x: Substring to match.
        lst: List of strings.

    Returns:
        The first matching string.
    """
    return [el for el in lst if x in el][0]

def box_constraint(box_type="LongOnly", lower=None, upper=None) -> dict:
    """
    Define a box constraint based on the type.

    Args:
        box_type (str): Constraint type; options: "LongOnly", "LongShort", or "Unbounded".
        lower: Lower bound(s).
        upper: Upper bound(s).

    Returns:
        dict: A dictionary with keys 'box_type', 'lower', and 'upper'.
    """
    box_type = match_arg(box_type, ["LongOnly", "LongShort", "Unbounded"])

    if box_type == "Unbounded":
        lower = float("-inf") if lower is None else lower
        upper = float("inf") if upper is None else upper
    elif box_type == "LongShort":
        lower = -1 if lower is None else lower
        upper = 1 if upper is None else upper
    elif box_type == "LongOnly":
        if lower is None:
            if upper is None:
                lower = 0
                upper = 1
            else:
                lower = 0  # lower set to 0 when only upper is provided
        else:
            if not np.isscalar(lower):
                if any(l < 0 for l in lower):
                    raise ValueError("Inconsistent lower bounds for box_type 'LongOnly'. "
                                     "Change box_type to LongShort or ensure that lower >= 0.")
            upper = lower * 0 + 1 if upper is None else upper

    return {'box_type': box_type, 'lower': lower, 'upper': upper}

def linear_constraint(Amat=None,
                      sense="=",
                      rhs=float("inf"),
                      index_or_name=None,
                      a_values=None) -> dict:
    """
    Create a dictionary representing a linear constraint.

    Args:
        Amat: Constraint matrix.
        sense (str): Constraint sense. Defaults to "=".
        rhs: Right-hand side value. Defaults to infinity.
        index_or_name: Optional index or name.
        a_values: Optional additional values.

    Returns:
        dict: A dictionary representing the linear constraint.
    """
    ans = {'Amat': Amat,
           'sense': sense,
           'rhs': rhs}
    if index_or_name is not None:
        ans['index_or_name'] = index_or_name
    if a_values is not None:
        ans['a_values'] = a_values
    return ans
