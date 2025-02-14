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

from typing import Union, Optional
import pandas as pd

class Selection:
    """
    A class to manage asset selection using filters and binary selection criteria.

    Attributes
    ----------
    selected : pd.Index
        The selected assets based on filtering criteria.
    _filtered : dict[str, Union[pd.Series, pd.DataFrame]]
        Dictionary storing filtering criteria as pandas Series or DataFrame.

    Methods
    -------
    get_selected(filter_names=None)
        Retrieves the selected asset indices based on the applied filters.
    clear()
        Resets the selection and clears all applied filters.
    add_filtered(filter_name, value)
        Adds a new filtering criterion to the selection.
    df(filter_names=None)
        Returns a DataFrame containing all applied filters.
    df_binary(filter_names=None)
        Returns a binary DataFrame where 1 indicates asset selection.
    """

    def __init__(self, ids: pd.Index = pd.Index([])):
        """
        Initializes the Selection object.

        Parameters
        ----------
        ids : pd.Index, optional
            The initial selection of asset indices, default is an empty index.
        """
        self._filtered: dict[str, Union[pd.Series, pd.DataFrame]] = {}
        self.selected = ids

    @property
    def selected(self) -> pd.Index:
        return self._selected

    @selected.setter
    def selected(self, value: pd.Index):
        if not isinstance(value, pd.Index):
            raise ValueError(
                "Inconsistent input type for selected.setter. Needs to be a pd.Index."
            )
        self._selected = value

    @property
    def filtered(self) -> dict[str, Union[pd.Series, pd.DataFrame]]:
        return self._filtered

    def get_selected(self, filter_names: Optional[list[str]] = None) -> pd.Index:
        """
        Retrieves the selected asset indices based on applied filters.

        Parameters
        ----------
        filter_names : list[str], optional
            List of filter names to consider. If None, all filters are used.

        Returns
        -------
        pd.Index
            The asset indices that satisfy all applied filters.
        """
        df = self.df_binary(filter_names) if filter_names is not None else self.df_binary()
        return df[df.eq(1).all(axis=1)].index

    def clear(self) -> None:
        """
        Clears all filters and resets the selection.

        Returns
        -------
        None
        """
        self.selected = pd.Index([])
        self._filtered = {}

    def add_filtered(self, filter_name: str, value: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Adds a new filtering criterion to the selection.

        Parameters
        ----------
        filter_name : str
            The name of the filter to be added.
        value : Union[pd.Series, pd.DataFrame]
            The filtering data as a pandas Series or DataFrame.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If filter_name is not a non-empty string.
            If value is not a pandas Series or DataFrame.
            If the 'binary' column contains values other than 0 or 1.
        """
        if not isinstance(filter_name, str) or not filter_name.strip():
            raise ValueError("Argument 'filter_name' must be a nonempty string.")

        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise ValueError(
                'Inconsistent input type. Needs to be a pd.Series or a pd.DataFrame.'
            )

        # Ensure that column 'binary' is of type int if it exists
        if isinstance(value, pd.Series):
            if value.name == 'binary' and not value.isin([0, 1]).all():
                raise ValueError("Column 'binary' must contain only 0s and 1s.")
            value = value.astype(int) if value.name == 'binary' else value

        if isinstance(value, pd.DataFrame) and 'binary' in value.columns:
            if not value['binary'].isin([0, 1]).all():
                raise ValueError("Column 'binary' must contain only 0s and 1s.")
            value['binary'] = value['binary'].astype(int)

        # Add to filtered
        self._filtered[filter_name] = value

        # Reset selected
        self.selected = self.get_selected()
        return None

    def df(self, filter_names: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame containing all applied filters.

        Parameters
        ----------
        filter_names : list[str], optional
            List of filter names to include. If None, all filters are used.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column represents a filter applied to the selection.
        """
        if filter_names is None:
            filter_names = self.filtered.keys()
        return pd.concat(
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

    def df_binary(self, filter_names: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Returns a binary DataFrame where 1 indicates an asset is selected.

        Parameters
        ----------
        filter_names : list[str], optional
            List of filter names to include. If None, all filters are used.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 1s indicating selected assets and 0s otherwise.
        """
        if filter_names is None:
            filter_names = self.filtered.keys()
        df = self.df(filter_names=filter_names).filter(like='binary').dropna()
        df.columns = df.columns.droplevel(1)
        return df
