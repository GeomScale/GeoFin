'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
### CLASS BacktestItemBuilder AND BACKTEST ITEM BUILDER FUNCTIONS
############################################################################

from typing import Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BacktestItemBuilder(ABC):
    """
    Abstract base class for building backtest items.

    This class provides a flexible framework for defining how items are built
    during backtesting, favoring flexibility over safety.

    Attributes
    ----------
    arguments : dict[str, Any]
        A dictionary of arguments used for item construction.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BacktestItemBuilder with provided arguments.

        Parameters
        ----------
        **kwargs :
            Key-value pairs of arguments for item building.
        """
        self._arguments = {}
        self._arguments.update(kwargs)

    @property
    def arguments(self) -> dict[str, Any]:
        """
        Returns the arguments dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary of arguments.
        """
        return self._arguments

    @arguments.setter
    def arguments(self, value: dict[str, Any]) -> None:
        """
        Sets the arguments dictionary.

        Parameters
        ----------
        value : dict[str, Any]
            A new dictionary of arguments.
        """
        self._arguments = value

    @abstractmethod
    def __call__(self, service, rebdate: str) -> None:
        """
        Abstract method to build the backtest item.

        Parameters
        ----------
        service : Any
            The backtest service instance.
        rebdate : str
            The rebalancing date.
        """
        raise NotImplementedError("Method '__call__' must be implemented in derived class.")


class SelectionItemBuilder(BacktestItemBuilder):
    """
    Builds selection items for backtesting based on a custom function.

    Methods
    -------
    __call__(bs, rebdate)
        Constructs and adds a selection item to the backtest service.
    """

    def __call__(self, bs, rebdate: str) -> None:
        """
        Constructs and adds a selection item to the backtest service.

        Parameters
        ----------
        bs : Any
            The backtest service instance.
        rebdate : str
            The rebalancing date.

        Raises
        ------
        ValueError
            If the custom function 'bibfn' is not defined or callable.
        """
        selection_item_builder_fn = self.arguments.get('bibfn')
        if selection_item_builder_fn is None or not callable(selection_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        item_value = selection_item_builder_fn(bs=bs, rebdate=rebdate, **self.arguments)
        item_name = self.arguments.get('item_name')

        # Add selection item
        bs.selection.add_filtered(filter_name=item_name, value=item_value)
        return None


class OptimizationItemBuilder(BacktestItemBuilder):
    """
    Builds optimization items for backtesting based on a custom function.

    Methods
    -------
    __call__(bs, rebdate)
        Constructs optimization data or constraints for the backtest service.
    """

    def __call__(self, bs, rebdate: str) -> None:
        """
        Constructs optimization data or constraints for the backtest service.

        Parameters
        ----------
        bs : Any
            The backtest service instance.
        rebdate : str
            The rebalancing date.

        Raises
        ------
        ValueError
            If the custom function 'bibfn' is not defined or callable.
        """
        optimization_item_builder_fn = self.arguments.get('bibfn')
        if optimization_item_builder_fn is None or not callable(optimization_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        # Call the custom function to modify the backtest service in place
        optimization_item_builder_fn(bs=bs, rebdate=rebdate, **self.arguments)
        return None


# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Selection
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.Series:
    """
    Filters assets based on minimum trading volume.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters including:
        - 'width': The rolling window width (default: 365).
        - 'agg_fn': Aggregation function applied to volume data (default: np.median).
        - 'min_volume': Minimum volume threshold (default: 500,000).

    Returns
    -------
    pd.Series
        A series of selected assets meeting the volume threshold.
    """
    width = kwargs.get('width', 365)
    agg_fn = kwargs.get('agg_fn', np.median)
    min_volume = kwargs.get('min_volume', 500_000)

    X_vol = (
        bs.data.get_volume_series(end_date=rebdate, width=width)
        .fillna(0).apply(agg_fn, axis=0)
    )

    ids = [col for col in X_vol.columns if agg_fn(X_vol[col]) >= min_volume]

    series = pd.Series(np.ones(len(ids)), index=ids, name='minimum_volume')
    bs.rebalancing.selection.add_filtered(filter_name=series.name, value=series)
    return None



def bibfn_selection_data(bs, rebdate: str, **kwargs) -> pd.Series:
    """
    Selects all available assets from the return series.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters.

    Returns
    -------
    pd.Series
        A binary series selecting all available assets.
    """
    data = bs.data.get('return_series')
    if data is None:
        raise ValueError('Return series data is missing.')

    return pd.Series(np.ones(data.shape[1], dtype=int), index=data.columns, name='binary')



def bibfn_selection_ltr(bs, rebdate: str, **kwargs) -> pd.DataFrame:
    """
    Defines the selection based on a Learn-to-Rank model.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters, including 'params_xgb' for XGBoost training.

    Returns
    -------
    pd.DataFrame
        DataFrame with scores and binary selections.
    """
  
    # Arguments
    params_xgb = kwargs.get('params_xgb')

    # Selection
    ids = bs.selection.selected

    # Extract data
    merged_df = bs.data.get('merged_df').copy()
    df_train = merged_df[merged_df['DATE'] < rebdate]#.reset_index(drop = True)
    df_test = merged_df[merged_df['DATE'] == rebdate]#.reset_index(drop = True)
    df_test = df_test[ df_test['ID'].isin(selected) ]
    ids = df_test['ID'].to_list()

    # Training data
    X_train = df_train.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
    y_train = df_train['label']
    grouped_train = df_train.groupby('DATE').size().to_numpy()
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtrain.set_group(grouped_train)

    # Evaluation data
    X_test = df_test.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
    grouped_test = df_test.groupby('DATE').size().to_numpy()
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(grouped_test)

    # Train and predict
    bst = xgb.train(params_xgb, dtrain, 100)
    scores = bst.predict(dtest) * (-1)

    # # Extract feature importance
    # f_importance = bst.get_score(importance_type='gain')

    return pd.DataFrame({'values': scores,
                         'binary': np.ones(len(scores), dtype = int),
                        }, index = scores.index)


# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Optimization data
# --------------------------------------------------------------------------

def bibfn_return_series(bs, rebdate: str, **kwargs) -> None:
    """
    Prepares single stock return series for optimization.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters, including:
        - 'width': int
            The rolling window size.

    Raises
    ------
    ValueError
        If the return series data is missing.

    Returns
    -------
    None
    """
    width = kwargs.get('width')

    ids = bs.selection.selected
    data = bs.data.get('return_series')
    if data is None:
        raise ValueError('Return series data is missing.')

    return_series = data[data.index <= rebdate].tail(width)[ids]
    return_series = return_series[return_series.index.dayofweek < 5]

    bs.optimization_data['return_series'] = return_series
    return None

def bibfn_bm_series(bs, rebdate: str, **kwargs) -> None:
    """
    Prepares benchmark series for optimization.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters, including:
        - 'width': int
            The rolling window width.
        - 'align': bool
            Whether to align the benchmark series with return series.

    Raises
    ------
    ValueError
        If the benchmark return series data is missing.

    Returns
    -------
    None
    """
    width = kwargs.get('width')
    align = kwargs.get('align')

    data = bs.data.get('bm_series')
    if data is None:
        raise ValueError('Benchmark return series data is missing.')

    bm_series = data[data.index <= rebdate].tail(width)
    bm_series = bm_series[bm_series.index.dayofweek < 5]

    bs.optimization_data['bm_series'] = bm_series

    if align:
        bs.optimization_data.align_dates(
            variable_names=['bm_series', 'return_series'],
            dropna=True
        )

    return None




# --------------------------------------------------------------------------
# Backtest item builder functions - Optimization constraints
# --------------------------------------------------------------------------

def bibfn_budget_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Sets the budget constraint for the optimization.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters, including 'budget' (default: 1).

    Returns
    -------
    None
    """
    budget = kwargs.get('budget', 1)
    bs.optimization.constraints.add_budget(rhs=budget, sense='=')
    return None

def bibfn_box_constraints(bs, rebdate: str, **kwargs) -> None:
    """
    Sets the box constraints for the optimization.

    Parameters
    ----------
    bs : Any
        The backtest service instance.
    rebdate : str
        The rebalancing date.
    **kwargs :
        Additional parameters, including:
        - 'lower': Lower bound (default: 0).
        - 'upper': Upper bound (default: 1).
        - 'box_type': Type of box constraint (default: 'LongOnly').

    Returns
    -------
    None
    """
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    box_type = kwargs.get('box_type', 'LongOnly')
    bs.optimization.constraints.add_box(box_type=box_type, lower=lower, upper=upper)
    return None




