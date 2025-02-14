
'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

import numpy as np
import pandas as pd
from helper_functions import to_numpy
from typing import Optional


class OptimizationData(dict):

    """
    A container for managing optimization-related data.

    This class extends the Python dictionary to support specific operations
    like aligning dates and handling lagged data for optimization tasks.

    Attributes
    ----------
    align : bool
        Whether to align dates across variables on initialization.
    lags : dict
        A dictionary specifying lag values for variables.
    """

    def __init__(self, align=True, lags={}, *args, **kwargs):

        """
        Initializes the OptimizationData instance.

        Parameters
        ----------
        align : bool, optional
            Whether to align dates across variables. Default is True.
        lags : dict, optional
            Dictionary specifying lags for variables. Keys are variable names
            and values are the lag amounts. Default is an empty dictionary.
        *args :
            Additional positional arguments for the dictionary.
        **kwargs :
            Additional keyword arguments for the dictionary.
        """

        super(OptimizationData, self).__init__(*args, **kwargs)
        self.__dict__ = self
        if len(lags) > 0:
            for key in lags.keys():
                self[key] = self[key].shift(lags[key])
        if align:
            self.align_dates()

    def align_dates(self, variable_names: Optional[list[str]] = None) -> None:

        """
        Aligns dates across the specified variables.

        Parameters
        ----------
        variable_names : list[str], optional
            List of variable names to align. If None, all variables are aligned.

        Returns
        -------
        None
        """

        if variable_names is None:
            variable_names = self.keys()
        index = self.intersecting_dates(variable_names=list(variable_names))
        for key in variable_names:
            self[key] = self[key].loc[index]

    def intersecting_dates(self,
                           variable_names: Optional[list[str]] = None,
                           dropna: bool = True) -> pd.DatetimeIndex:
        
        """
        Finds the intersection of dates across the specified variables.

        Parameters
        ----------
        variable_names : list[str], optional
            List of variable names to find intersecting dates for. If None, all variables are used.
        dropna : bool, optional
            Whether to drop rows with NaN values in the variables. Default is True.

        Returns
        -------
        pd.DatetimeIndex
            The intersection of dates across the specified variables.
        """

        if variable_names is None:
            variable_names = list(self.keys())
        if dropna:
            for variable_name in variable_names:
                self[variable_name] = self[variable_name].dropna()
        index = self.get(variable_names[0]).index
        for variable_name in variable_names:
            index = index.intersection(self.get(variable_name).index)
        return index
    
    