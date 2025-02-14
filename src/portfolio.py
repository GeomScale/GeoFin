'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''

import pandas as pd
import numpy as np

class Portfolio:
    """
    A class representing a financial portfolio with rebalancing capabilities.

    Attributes
    ----------
    rebalancing_date : str
        The date when the portfolio is rebalanced.
    weights : dict
        A dictionary representing asset weights in the portfolio.
    name : str
        The name of the portfolio.
    init_weights : dict
        Initial weights of the portfolio before rebalancing.
    """

    def __init__(self,
                 rebalancing_date: str = None,
                 weights: dict = {},
                 name: str = None,
                 init_weights: dict = {}):
        """
        Initializes a Portfolio instance.

        Parameters
        ----------
        rebalancing_date : str, optional
            The date of rebalancing, by default None.
        weights : dict, optional
            Asset weights in the portfolio, by default an empty dictionary.
        name : str, optional
            The name of the portfolio, by default None.
        init_weights : dict, optional
            Initial asset weights before rebalancing, by default an empty dictionary.
        """
        self.rebalancing_date = rebalancing_date
        self.weights = weights
        self.name = name
        self.init_weights = init_weights

    @staticmethod
    def empty() -> 'Portfolio':
        return Portfolio()

    @property
    def weights(self):
        return self._weights

    def get_weights_series(self) -> pd.Series:
        return pd.Series(self._weights)

    @weights.setter
    def weights(self, new_weights: dict):

        if not isinstance(new_weights, dict):
            if hasattr(new_weights, 'to_dict'):
                new_weights = new_weights.to_dict()
            else:
                raise TypeError('weights must be a dictionary')
        self._weights = new_weights

    @property
    def rebalancing_date(self):
        return self._rebalancing_date

    @rebalancing_date.setter
    def rebalancing_date(self, new_date: str):
        if new_date and not isinstance(new_date, str):
            raise TypeError('date must be a string')
        self._rebalancing_date = new_date

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError('name must be a string')
        self._name = new_name

    def __repr__(self):
        """
        Returns a string representation of the Portfolio object.

        Returns
        -------
        str
            String representation of the portfolio.
        """
        return f'Portfolio(rebalancing_date={self.rebalancing_date}, weights={self.weights})'

    def float_weights(self,
                      return_series: pd.DataFrame,
                      end_date: str,
                      rescale: bool = False) -> pd.DataFrame:
        """
        Computes the floating weights of the portfolio over time.

        Parameters
        ----------
        return_series : pd.DataFrame
            A DataFrame containing asset return data indexed by date.
        end_date : str
            The ending date for computing floating weights.
        rescale : bool, optional
            Whether to rescale the weights such that their sum remains 1, by default False.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of floating weights over time, or None if weights are not set.
        """
        if self.weights is not None:
            return floating_weights(X=return_series,
                                    w=self.weights,
                                    start_date=self.rebalancing_date,
                                    end_date=end_date,
                                    rescale=rescale)
        else:
            return None

    def initial_weights(self,
                        selection: list[str],
                        return_series: pd.DataFrame,
                        end_date: str,
                        rescale: bool = True) -> dict[str, float]:
        """
        Computes the initial weights of the portfolio at the rebalancing date.

        Parameters
        ----------
        selection : list[str]
            List of asset names to include in the initial weights.
        return_series : pd.DataFrame
            A DataFrame containing asset return data indexed by date.
        end_date : str
            The ending date for computing the weights.
        rescale : bool, optional
            Whether to rescale the weights to sum to 1, by default True.

        Returns
        -------
        dict[str, float]
            Dictionary containing the initial asset weights.

        Notes
        -----
        - If `self.rebalancing_date` and `self.weights` are set, the function calculates
          the weights by floating them to the end date.
        - If these attributes are not set, it returns None.
        """
        if not hasattr(self, '_initial_weights'):
            if self.rebalancing_date is not None and self.weights is not None:
                w_init = dict.fromkeys(selection, 0)
                w_float = self.float_weights(return_series=return_series,
                                             end_date=end_date,
                                             rescale=rescale)
                w_floated = w_float.iloc[-1]

                w_init.update({key: w_floated[key] for key in w_init.keys() & w_floated.keys()})
                self._initial_weights = w_init
            else:
                self._initial_weights = None  # {key: 0 for key in selection}

        return self._initial_weights

    def turnover(self, portfolio: "Portfolio", return_series: pd.DataFrame, rescale: bool = True) -> float:
        """
        Computes the portfolio turnover by comparing the previous and current portfolio weights.

        Parameters
        ----------
        portfolio : Portfolio
            The previous portfolio to compare against.
        return_series : pd.DataFrame
            A DataFrame containing asset return data indexed by date.
        rescale : bool, optional
            Whether to rescale the weights to sum to 1, by default True.

        Returns
        -------
        float
            The total absolute turnover of the portfolio.

        Notes
        -----
        - Turnover measures the total change in portfolio weights between two consecutive rebalancing dates.
        - If `portfolio.rebalancing_date` is before `self.rebalancing_date`, it uses `portfolio.initial_weights()`.
        - Otherwise, it computes initial weights from `self.initial_weights()`.
        """
        if portfolio.rebalancing_date is not None and portfolio.rebalancing_date < self.rebalancing_date:
            w_init = portfolio.initial_weights(selection=self.weights.keys(),
                                               return_series=return_series,
                                               end_date=self.rebalancing_date,
                                               rescale=rescale)
        else:
            w_init = self.initial_weights(selection=portfolio.weights.keys(),
                                          return_series=return_series,
                                          end_date=portfolio.rebalancing_date,
                                          rescale=rescale)

        return pd.Series(w_init).sub(pd.Series(portfolio.weights), fill_value=0).abs().sum()



class Strategy:
    """
    A class representing a financial trading strategy consisting of multiple portfolios.

    Attributes
    ----------
    portfolios : list[Portfolio]
        A list of Portfolio objects in the strategy.
    """

    def __init__(self, portfolios: list[Portfolio]):
        """
        Initializes a Strategy instance.

        Parameters
        ----------
        portfolios : list[Portfolio]
            A list of Portfolio objects.
        """
        self.portfolios = portfolios

    @property
    def portfolios(self):
        return self._portfolios

    @portfolios.setter
    def portfolios(self, new_portfolios: list[Portfolio]):
        if not isinstance(new_portfolios, list):
            raise TypeError('portfolios must be a list')
        if not all(isinstance(portfolio, Portfolio) for portfolio in new_portfolios):
            raise TypeError('all elements in portfolios must be of type Portfolio')
        self._portfolios = new_portfolios
    

    def clear(self) -> None:
        """
        Clears the portfolio list.

        Returns
        -------
        None
        """
        self.portfolios.clear()
        return None

    def get_rebalancing_dates(self):
        """
        Retrieves all portfolio rebalancing dates.

        Returns
        -------
        list[str]
            List of rebalancing dates.
        """
        return [portfolio.rebalancing_date for portfolio in self.portfolios]

    def get_weights_df(self) -> pd.DataFrame:
        """
        Returns portfolio weights as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing portfolio weights.
        """
        weights_dict = {}
        for portfolio in self.portfolios:
            weights_dict[portfolio.rebalancing_date] = portfolio.weights
        return pd.DataFrame(weights_dict).T


    def get_portfolio(self, rebalancing_date: str) -> Portfolio:
        """
        Retrieves a portfolio for a specific rebalancing date.

        Parameters
        ----------
        rebalancing_date : str
            The date for which the portfolio is requested.

        Returns
        -------
        Portfolio
            The corresponding portfolio.

        Raises
        ------
        ValueError
            If no portfolio is found for the specified date.
        """
        if rebalancing_date in self.get_rebalancing_dates():
            idx = self.get_rebalancing_dates().index(rebalancing_date)
            return self.portfolios[idx]
        else:
            raise ValueError(f'No portfolio found for rebalancing date {rebalancing_date}')

    def has_previous_portfolio(self, rebalancing_date: str) -> bool:
        """
        Checks whether a previous portfolio exists before a given rebalancing date.

        Parameters
        ----------
        rebalancing_date : str
            The reference date.

        Returns
        -------
        bool
            True if there is a previous portfolio, otherwise False.
        """
        dates = self.get_rebalancing_dates()
        return len(dates) > 0 and dates[0] < rebalancing_date

    def get_previous_portfolio(self, rebalancing_date: str) -> Portfolio:
        """
        Retrieves the most recent portfolio before a given rebalancing date.

        Parameters
        ----------
        rebalancing_date : str
            The reference date.

        Returns
        -------
        Portfolio
            The previous portfolio, or an empty portfolio if none exist.
        """
        if not self.has_previous_portfolio(rebalancing_date):
            return Portfolio.empty()
        else:
            previous_date = [x for x in self.get_rebalancing_dates() if x < rebalancing_date][-1]
            return self.get_portfolio(previous_date)

    def get_initial_portfolio(self, rebalancing_date: str) -> Portfolio:
        """
        Retrieves the initial portfolio before the specified rebalancing date.

        Parameters
        ----------
        rebalancing_date : str
            The reference date.

        Returns
        -------
        Portfolio
            The initial portfolio before the given date, or an empty portfolio if none exist.
        """
        if self.has_previous_portfolio(rebalancing_date=rebalancing_date):
            return self.get_previous_portfolio(rebalancing_date)
        else:
            return Portfolio(rebalancing_date=None, weights={})

    def __repr__(self) -> str:
        """
        Returns a string representation of the Strategy object.

        Returns
        -------
        str
            String representation of the strategy.
        """
        return f'Strategy(portfolios={self.portfolios})'

    def number_of_assets(self, th: float = 0.0001) -> pd.Series:
        """
        Computes the number of assets in each portfolio above a given threshold.

        Parameters
        ----------
        th : float, optional
            The minimum absolute weight to consider an asset as included, by default 0.0001.

        Returns
        -------
        pd.Series
            Series containing the number of assets per rebalancing date.
        """
        return self.get_weights_df().apply(lambda x: sum(np.abs(x) > th), axis=1)

    def turnover(self, return_series: pd.DataFrame, rescale: bool = True) -> pd.Series:
        """
        Computes the turnover for each rebalancing period.

        Parameters
        ----------
        return_series : pd.DataFrame
            A DataFrame containing asset return data indexed by date.
        rescale : bool, optional
            Whether to rescale the weights, by default True.

        Returns
        -------
        pd.Series
            Series of turnover values indexed by rebalancing dates.
        """
        dates = self.get_rebalancing_dates()
        turnover = {}
        for rebalancing_date in dates:
            previous_portfolio = self.get_previous_portfolio(rebalancing_date)
            current_portfolio = self.get_portfolio(rebalancing_date)
            turnover[rebalancing_date] = current_portfolio.turnover(portfolio=previous_portfolio,
                                                                    return_series=return_series,
                                                                    rescale=rescale)
        return pd.Series(turnover)

    def simulate(self,
                 return_series: pd.DataFrame = None,
                 fc: float = 0,
                 vc: float = 0,
                 n_days_per_year: int = 252) -> pd.Series:
        """
        Simulates portfolio performance over time, incorporating fixed and variable transaction costs.

        Parameters
        ----------
        return_series : pd.DataFrame
            DataFrame containing asset return data indexed by date.
        fc : float, optional
            Fixed transaction cost per rebalancing, by default 0.
        vc : float, optional
            Variable transaction cost proportional to turnover, by default 0.
        n_days_per_year : int, optional
            Number of trading days per year, by default 252.

        Returns
        -------
        pd.Series
            A series of portfolio returns over time.
        """
        rebdates = self.get_rebalancing_dates()
        ret_list = []
        for rebdate in rebdates:
            next_rebdate = rebdates[rebdates.index(rebdate) + 1] if rebdate < rebdates[-1] else return_series.index[-1]

            portfolio = self.get_portfolio(rebdate)
            w_float = portfolio.float_weights(return_series=return_series,
                                              end_date=next_rebdate,
                                              rescale=False)  # Rescale is hardcoded to False.
            short_positions = list(filter(lambda x: x < 0, portfolio.weights.values()))
            long_positions = list(filter(lambda x: x >= 0, portfolio.weights.values()))
            margin = abs(sum(short_positions))
            cash = max(min(1 - sum(long_positions), 1), 0)
            loan = 1 - (sum(long_positions) + cash) - (sum(short_positions) + margin)
            w_float.insert(0, 'margin', margin)
            w_float.insert(0, 'cash', cash)
            w_float.insert(0, 'loan', loan)
            level = w_float.sum(axis=1)
            ret_tmp = level.pct_change(1)  # 1-day lookback
            ret_list.append(ret_tmp)

        portf_ret = pd.concat(ret_list).dropna()

        if vc != 0:
            to = self.turnover(return_series=return_series,
                               rescale=False)  # Rescale is hardcoded to False.
            varcost = to * vc
            portf_ret.iloc[0] -= varcost.iloc[0]
            portf_ret.iloc[1:] -= varcost.iloc[1:].values

        if fc != 0:
            n_days = (portf_ret.index[1:] - portf_ret.index[:-1]).to_numpy().astype('timedelta64[D]').astype(int)
            fixcost = (1 + fc) ** (n_days / n_days_per_year) - 1
            portf_ret.iloc[1:] -= fixcost

        return portf_ret


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def floating_weights(X, w, start_date, end_date, rescale=True):
    """
    Computes floating weights over a time period given initial weights.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing asset returns.
    w : dict
        Dictionary of initial asset weights.
    start_date : str
        Start date for weight computation.
    end_date : str
        End date for weight computation.
    rescale : bool, optional
        Whether to rescale weights to sum to 1, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame of floating weights.

    Raises
    ------
    ValueError
        If start_date or end_date are not contained in the dataset.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date < X.index[0]:
        raise ValueError('start_date must be contained in dataset')
    if end_date > X.index[-1]:
        raise ValueError('end_date must be contained in dataset')

    w = pd.Series(w, index=w.keys())
    if w.isna().any():
        raise ValueError('weights (w) contain NaN which is not allowed.')
    else:
        w = w.to_frame().T
    xnames = X.columns
    wnames = w.columns

    if not all(wnames.isin(xnames)):
        raise ValueError('Not all assets in w are contained in X.')

    X_tmp = X.loc[start_date:end_date, wnames].copy().fillna(0)
    xmat = 1 + X_tmp
    xmat.iloc[0] = w.dropna(how='all').fillna(0)
    w_float = xmat.cumprod()

    if rescale:
        w_float_long = w_float.where(w_float >= 0).div(w_float[w_float >= 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float_short = w_float.where(w_float < 0).div(w_float[w_float < 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float = pd.DataFrame(w_float_long + w_float_short, index=xmat.index, columns=wnames)

    return w_float
