'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### CLASS Selection
############################################################################



from typing import Union, Optional, List
import pandas as pd

class Selection:
    def __init__(self, ids: pd.Index = pd.Index([])):
        self._filtered: dict[str, Union[pd.Series, pd.DataFrame]] = {}
        self.selected = ids

    @property
    def selected(self) -> pd.Index:
        return self._selected

    @selected.setter
    def selected(self, value: pd.Index):
        if not isinstance(value, pd.Index):
            raise ValueError("Inconsistent input type for selected.setter. Needs to be a pd.Index.")
        self._selected = value

    @property
    def filtered(self):
        return self._filtered

    def get_selected(self, filter_names: Optional[List[str]] = None) -> pd.Index:
        if filter_names is not None:
            df = self.df_binary(filter_names)
        else:
            df = self.df_binary()
        #return only rows where all binary columns equal 1
        return df[df.eq(1).all(axis=1)].index

    def clear(self) -> None:
        self.selected = pd.Index([])
        self._filtered = {}

    def add_filtered(self, filter_name: str, value: Union[pd.Series, pd.DataFrame]) -> None:
        #check input types
        if not isinstance(filter_name, str) or not filter_name.strip():
            raise ValueError("Argument 'filter_name' must be a nonempty string.")

        if not isinstance(value, pd.Series) and not isinstance(value, pd.DataFrame):
            raise ValueError("Inconsistent input type. Needs to be a pd.Series or a pd.DataFrame.")

        # Ensure that column 'binary' is of type int if it exists
        if isinstance(value, pd.Series):
            if value.name == 'binary':
                if not value.isin([0, 1]).all():
                    raise ValueError("Column 'binary' must contain only 0s and 1s.")
                value = value.astype(int)
        elif isinstance(value, pd.DataFrame):
            if 'binary' in value.columns:
                if not value['binary'].isin([0, 1]).all():
                    raise ValueError("Column 'binary' must contain only 0s and 1s.")
                value['binary'] = value['binary'].astype(int)

        #add to filtered
        self._filtered[filter_name] = value

        #reset selected based on the updated filters
        self.selected = self.get_selected()

    def df(self, filter_names: Optional[List[str]] = None) -> pd.DataFrame:
        if filter_names is None:
            filter_names = list(self.filtered.keys())
        #if there are no filters, return an empty DataFrame
        if not filter_names:
            return pd.DataFrame()
        df_concat = pd.concat(
            {
                key: (
                    pd.DataFrame(self.filtered[key])
                    if isinstance(self.filtered[key], pd.Series)
                    else self.filtered[key]
                )
                for key in filter_names
            },
            axis=1,
        )
        return df_concat

    def df_binary(self, filter_names: Optional[List[str]] = None) -> pd.DataFrame:
        if filter_names is None:
            filter_names = list(self.filtered.keys())
        if not filter_names:
            return pd.DataFrame()
        df = self.df(filter_names=filter_names).filter(like='binary').dropna()
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        return df

