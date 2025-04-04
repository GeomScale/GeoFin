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
    def __init__(self,
                 rebalancing_date: str = None,
                 weights: dict = None,
                 name: str = None,
                 init_weights: dict = None):
        self.rebalancing_date = rebalancing_date
        self.weights = weights if weights is not None else {}
        self.name = name
        self.init_weights = init_weights if init_weights is not None else {}

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
        return f'Portfolio(rebalancing_date={self.rebalancing_date}, weights={self.weights})'

    def float_weights(self,
                      return_series: pd.DataFrame,
                      end_date: str,
                      rescale: bool = False):
        if self.weights is not None and self.weights != {}:
            return floating_weights(
                X=return_series,
                w=self.weights,
                start_date=self.rebalancing_date,
                end_date=end_date,
                rescale=rescale
            )
        else:
            return None

    def initial_weights(self,
                        selection: list[str],
                        return_series: pd.DataFrame,
                        end_date: str,
                        rescale: bool = True) -> dict[str, float]:
        if not hasattr(self, '_initial_weights'):
            if self.rebalancing_date is not None and self.weights:
                w_init = dict.fromkeys(selection, 0)
                w_float = self.float_weights(
                    return_series=return_series,
                    end_date=end_date,
                    rescale=rescale
                )
                if w_float is None or w_float.empty:
                    self._initial_weights = None
                else:
                    # Use intersection of keys safely by converting to sets.
                    common_keys = set(w_init.keys()).intersection(set(w_float.columns))
                    w_floated = w_float.iloc[-1]
                    w_init.update({key: w_floated[key] for key in common_keys})
                    self._initial_weights = w_init
            else:
                self._initial_weights = None
        return self._initial_weights

    def turnover(self, portfolio: "Portfolio", return_series: pd.DataFrame, rescale=True):
        # Fix: Compare dates as datetime objects rather than as raw strings.
        date_self = pd.to_datetime(self.rebalancing_date) if self.rebalancing_date else None
        date_other = pd.to_datetime(portfolio.rebalancing_date) if portfolio.rebalancing_date else None

        if date_other is not None and date_self is not None and date_other < date_self:
            w_init = portfolio.initial_weights(
                selection=list(self.weights.keys()),
                return_series=return_series,
                end_date=self.rebalancing_date,
                rescale=rescale
            )
        else:
            w_init = self.initial_weights(
                selection=list(portfolio.weights.keys()),
                return_series=return_series,
                end_date=portfolio.rebalancing_date,
                rescale=rescale
            )
        if w_init is None:
            return 0.0
        return pd.Series(w_init).sub(pd.Series(portfolio.weights), fill_value=0).abs().sum()


class Strategy:
    def __init__(self, portfolios: list[Portfolio]):
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
        self.portfolios.clear()

    def get_rebalancing_dates(self):
        return [portfolio.rebalancing_date for portfolio in self.portfolios]

    def get_weights(self, rebalancing_date: str) -> dict[str, float]:
        for portfolio in self.portfolios:
            if portfolio.rebalancing_date == rebalancing_date:
                return portfolio.weights
        return None

    def get_weights_df(self) -> pd.DataFrame:
        weights_dict = {}
        for portfolio in self.portfolios:
            weights_dict[portfolio.rebalancing_date] = portfolio.weights
        return pd.DataFrame(weights_dict).T

    def get_portfolio(self, rebalancing_date: str) -> Portfolio:
        dates = self.get_rebalancing_dates()
        if rebalancing_date in dates:
            idx = dates.index(rebalancing_date)
            return self.portfolios[idx]
        else:
            raise ValueError(f'No portfolio found for rebalancing date {rebalancing_date}')

    def has_previous_portfolio(self, rebalancing_date: str) -> bool:
        dates = self.get_rebalancing_dates()
        if dates:
            # Compare as datetime objects for safety.
            first_date = pd.to_datetime(dates[0])
            curr_date = pd.to_datetime(rebalancing_date)
            return first_date < curr_date
        return False

    def get_previous_portfolio(self, rebalancing_date: str) -> Portfolio:
        if not self.has_previous_portfolio(rebalancing_date):
            return Portfolio.empty()
        else:
            previous_dates = [x for x in self.get_rebalancing_dates() if pd.to_datetime(x) < pd.to_datetime(rebalancing_date)]
            previous_date = sorted(previous_dates)[-1]
            return self.get_portfolio(previous_date)

    def get_initial_portfolio(self, rebalancing_date: str) -> Portfolio:
        if self.has_previous_portfolio(rebalancing_date=rebalancing_date):
            initial_portfolio = self.get_previous_portfolio(rebalancing_date)
        else:
            initial_portfolio = Portfolio(rebalancing_date=None, weights={})
        return initial_portfolio

    def __repr__(self):
        return f'Strategy(portfolios={self.portfolios})'

    def number_of_assets(self, th: float = 0.0001) -> pd.Series:
        return self.get_weights_df().apply(lambda x: sum(np.abs(x) > th), axis=1)

    def turnover(self, return_series, rescale=True) -> pd.Series:
        dates = self.get_rebalancing_dates()
        turnover_dict = {}
        for r_date in dates:
            previous_portfolio = self.get_previous_portfolio(r_date)
            current_portfolio = self.get_portfolio(r_date)
            turnover_dict[r_date] = current_portfolio.turnover(
                portfolio=previous_portfolio,
                return_series=return_series,
                rescale=rescale
            )
        return pd.Series(turnover_dict)

    def simulate(self,
                 return_series: pd.DataFrame = None,
                 fc: float = 0,
                 vc: float = 0,
                 n_days_per_year: int = 252) -> pd.Series:
        rebdates = self.get_rebalancing_dates()
        ret_list = []
        for i, rebdate in enumerate(rebdates):
            # Fix: Compare dates as datetimes.
            if i < len(rebdates) - 1:
                next_rebdate = rebdates[i + 1]
            else:
                next_rebdate = return_series.index[-1]

            portfolio = self.get_portfolio(rebdate)
            w_float = portfolio.float_weights(
                return_series=return_series,
                end_date=next_rebdate,
                rescale=False
            )
            short_positions = list(filter(lambda x: x < 0, portfolio.weights.values()))
            long_positions = list(filter(lambda x: x >= 0, portfolio.weights.values()))
            margin = abs(sum(short_positions))
            cash = max(min(1 - sum(long_positions), 1), 0)
            loan = 1 - (sum(long_positions) + cash) - (sum(short_positions) + margin)

            w_float.insert(0, 'margin', margin)
            w_float.insert(0, 'cash', cash)
            w_float.insert(0, 'loan', loan)
            level = w_float.sum(axis=1)
            ret_tmp = level.pct_change(1)  # One day lookback.
            ret_list.append(ret_tmp)

        portf_ret = pd.concat(ret_list).dropna()

        if vc != 0:
            to = self.turnover(return_series=return_series, rescale=False)
            varcost = to * vc
            portf_ret = portf_ret.subtract(varcost, fill_value=0)

        if fc != 0:
            date_index = pd.to_datetime(portf_ret.index)
            n_days = (date_index[1:] - date_index[:-1]).days
            fixcost = (1 + fc) ** (np.array(n_days) / n_days_per_year) - 1
            fixcost_series = pd.Series(fixcost, index=portf_ret.index[1:])
            portf_ret.loc[fixcost_series.index] = portf_ret.loc[fixcost_series.index] - fixcost_series

        return portf_ret


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def floating_weights(X: pd.DataFrame, w: dict, start_date, end_date, rescale=True):
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
    # TODO : To extend to short positions cases when the weights can be negative
    # short_positions = wnames[w.iloc[0,:] < 0 ]
    # if len(short_positions) > 0:
    #     X_tmp[short_positions] = X_tmp[short_positions] * (-1)
    xmat = 1 + X_tmp
    # xmat.iloc[0] = w.dropna(how='all').fillna(0).abs()
    xmat.iloc[0] = w.dropna(how='all').fillna(0)
    w_float = xmat.cumprod()

    if rescale:
        w_float_long = w_float.where(w_float >= 0).div(w_float[w_float >= 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float_short = w_float.where(w_float < 0).div(w_float[w_float < 0].abs().sum(axis=1), axis='index').fillna(0)
        w_float = pd.DataFrame(w_float_long + w_float_short, index=xmat.index, columns=wnames)

    return w_float
