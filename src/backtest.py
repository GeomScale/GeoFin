'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

############################################################################
### CLASSES BacktestData, BacktestService, Backtest
############################################################################

import os
from typing import Optional
import pickle

import pandas as pd

from optimization import Optimization, EmptyOptimization
from optimization_data import OptimizationData
from constraints import Constraints
from portfolio import Portfolio, Strategy
from selection import Selection
from builders import SelectionItemBuilder, OptimizationItemBuilder


class BacktestData:
    """
    Represents the data required for backtesting.

    This class acts as a container for any data-related requirements for backtesting.

    Attributes
    ----------
    None
    """

    def __init__(self):
        pass


class BacktestService:
    """
    Manages backtesting services, including selection, optimization, and settings.

    Attributes
    ----------
    data : BacktestData
        The backtest data.
    selection_item_builders : dict[str, SelectionItemBuilder]
        Builders for selection items.
    optimization_item_builders : dict[str, OptimizationItemBuilder]
        Builders for optimization items.
    optimization : Optional[Optimization]
        The optimization instance. Defaults to `EmptyOptimization`.
    settings : Optional[dict]
        Additional settings for the backtest.
    """

    def __init__(self,
                 data: BacktestData,
                 selection_item_builders: dict[str, SelectionItemBuilder],
                 optimization_item_builders: dict[str, OptimizationItemBuilder],
                 optimization: Optional[Optimization] = EmptyOptimization(),
                 settings: Optional[dict] = None,
                 **kwargs) -> None:
        """
        Initializes the BacktestService class.

        Parameters
        ----------
        data : BacktestData
            The backtest data.
        selection_item_builders : dict
            Dictionary of selection item builders.
        optimization_item_builders : dict
            Dictionary of optimization item builders.
        optimization : Optional[Optimization], optional
            Optimization instance, by default EmptyOptimization().
        settings : Optional[dict], optional
            Additional settings, by default None.
        **kwargs :
            Additional settings.
        """
        self.data = data
        self.optimization = optimization
        self.selection_item_builders = selection_item_builders
        self.optimization_item_builders = optimization_item_builders
        self.settings = settings if settings is not None else {}
        self.settings.update(kwargs)
        self.selection = Selection()
        self.optimization_data = OptimizationData([])


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        if not isinstance(value, Selection):
            raise TypeError("Expected a Selection instance for 'selection'")
        self._selection = value

    @property
    def selection_item_builders(self):
        return self._selection_item_builders

    @selection_item_builders.setter
    def selection_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, SelectionItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing SelectionItemBuilder instances "
                "for 'selection_item_builders'"
            )
        self._selection_item_builders = value

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if not isinstance(value, Optimization):
            raise TypeError("Expected an Optimization instance for 'optimization'")
        self._optimization = value

    @property
    def optimization_item_builders(self):
        return self._optimization_item_builders

    @optimization_item_builders.setter
    def optimization_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, OptimizationItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing OptimizationItemBuilder instances "
                "for 'optimization_item_builders'"
            )
        self._optimization_item_builders = value

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        if not isinstance(value, dict):
            raise TypeError("Expected a dictionary for 'settings'")
        self._settings = value



    def build_selection(self, rebdate: str) -> None:
        """
        Builds the selection process for a given rebalancing date.

        Parameters
        ----------
        rebdate : str
            The rebalancing date.
        """
        for key, item_builder in self.selection_item_builders.items():
            item_builder.arguments['item_name'] = key
            item_builder(self, rebdate)
        return None

    def build_optimization(self, rebdate: str) -> None:
        """
        Builds the optimization problem for a given rebalancing date.

        Parameters
        ----------
        rebdate : str
            The rebalancing date.
        """
        self.optimization.constraints = Constraints(selection=self.selection.selected)
        for item_builder in self.optimization_item_builders.values():
            item_builder(self, rebdate)
        return None

    def prepare_rebalancing(self, rebalancing_date: str) -> None:
        """
        Prepares the selection and optimization for a rebalancing date.

        Parameters
        ----------
        rebalancing_date : str
            The rebalancing date.
        """
        self.build_selection(rebdate=rebalancing_date)
        self.build_optimization(rebdate=rebalancing_date)
        return None


class Backtest:
    """
    Performs portfolio backtesting, including strategy building and output storage.

    Attributes
    ----------
    strategy : Strategy
        The backtesting strategy.
    output : dict
        The backtesting output.
    """

    def __init__(self) -> None:
        """
        Initializes the Backtest class.
        """
        self._strategy = Strategy([])
        self._output = {}

    @property
    def strategy(self):
        return self._strategy

    @property
    def output(self):
        return self._output
    

    def append_output(self, date_key=None, output_key=None, value=None):
        """
        Appends output data for a specific date and output key.

        Parameters
        ----------
        date_key : str, optional
            The date key for the output.
        output_key : str, optional
            The output key.
        value : any, optional
            The value to append.
        """
        if value is None:
            return True
        if date_key in self.output.keys():
            if output_key in self.output[date_key].keys():
                raise Warning(f"Output key '{output_key}' for date key '{date_key}' already exists and will be overwritten.")
            self.output[date_key][output_key] = value
        else:
            self.output[date_key] = {}
            self.output[date_key].update({output_key: value})
        return True

    def rebalance(self, bs: BacktestService, rebalancing_date: str) -> None:
        """
        Performs portfolio rebalancing for a given date.

        Parameters
        ----------
        bs : BacktestService
            The backtesting service instance.
        rebalancing_date : str
            The rebalancing date.
        """
        bs.prepare_rebalancing(rebalancing_date=rebalancing_date)
        try:
            bs.optimization.set_objective(optimization_data=bs.optimization_data)
            bs.optimization.solve()
        except Exception as error:
            raise RuntimeError(error)
        return None

    def run(self, bs: BacktestService) -> None:
        """
        Executes the backtest for all rebalancing dates.

        Parameters
        ----------
        bs : BacktestService
            The backtesting service instance.
        """
        for rebalancing_date in bs.settings['rebdates']:
            if not bs.settings.get('quiet'):
                print(f'Rebalancing date: {rebalancing_date}')
            self.rebalance(bs=bs, rebalancing_date=rebalancing_date)
            weights = bs.optimization.results['weights']
            portfolio = Portfolio(rebalancing_date=rebalancing_date, weights=weights)
            self.strategy.portfolios.append(portfolio)
            append_fun = bs.settings.get('append_fun')
            if append_fun is not None:
                append_fun(backtest=self, bs=bs, rebalancing_date=rebalancing_date, what=bs.settings.get('append_fun_args'))
        return None

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Saves the backtest object to a file.

        Parameters
        ----------
        filename : str
            The filename for the output file.
        path : Optional[str], optional
            The path where the file should be saved.
        """
        try:
            if path is not None and filename is not None:
                filename = os.path.join(path, filename)
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object:", ex)
        return None


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def append_custom(backtest: Backtest,
                  bs: BacktestService,
                  rebalancing_date: Optional[str] = None,
                  what: Optional[list] = None) -> None:
    """
    Appends custom data to the backtest output.

    Parameters
    ----------
    backtest : Backtest
        The backtest instance.
    bs : BacktestService
        The backtesting service instance.
    rebalancing_date : Optional[str], optional
        The rebalancing date.
    what : Optional[list], optional
        List of output keys to append.
    """
    if what is None:
        what = ['w_dict', 'objective']
    for key in what:
        if key == 'w_dict':
            w_dict = bs.optimization.results['w_dict']
            for key in w_dict.keys():
                weights = w_dict[key]
                if hasattr(weights, 'to_dict'):
                    weights = weights.to_dict()
                portfolio = Portfolio(rebalancing_date=rebalancing_date, weights=weights)
                backtest.append_output(date_key=rebalancing_date,
                                        output_key=f'weights_{key}',
                                        value=pd.Series(portfolio.weights))
        else:
            if not key in bs.optimization.results.keys():
                continue
            backtest.append_output(date_key=rebalancing_date,
                                    output_key=key,
                                    value=bs.optimization.results[key])
    return None


