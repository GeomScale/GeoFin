

'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''


import os
from typing import Optional, Union, Any
import pandas as pd
import pickle


def load_pickle(filename: str, path: Optional[str] = None) -> Union[Any, None]:

    """
    Loads a Python object from a pickle file.

    Parameters
    ----------
    filename : str
        The name of the pickle file to load.
    path : Optional[str], optional
        The directory path to the file. If not provided, only the filename is used.

    Returns
    -------
    Any
        The object loaded from the pickle file, or None if an error occurs.

    Raises
    ------
    EOFError
        If the file is empty or corrupted.
    Exception
        For other errors encountered during the unpickling process.
    """

    if path is not None:
        filename = os.path.join(path, filename)
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except EOFError:
        print("Error: Ran out of input. The file may be empty or corrupted.")
        return None
    except Exception as ex:
        print("Error during unpickling object:", ex)
    return None



def load_data_msci(path: str = None, n: int = 24) -> dict[str, pd.DataFrame]:

    """
    Loads MSCI daily returns data and benchmark series.

    Parameters
    ----------
    path : str, optional
        The directory containing the data files. If None, defaults to the `data` folder
        in the current working directory.
    n : int, optional
        The number of MSCI country indices to load. Default is 24.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary with the following keys:
        - 'return_series': A DataFrame containing the MSCI country index return series.
        - 'bm_series': A DataFrame containing the MSCI World index return series.

    Raises
    ------
    FileNotFoundError
        If the required data files are not found in the specified path.
    ValueError
        If the data files have unexpected formatting or contents.
    """

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path
    # Load MSCI country index return series
    try:
        df = pd.read_csv(os.path.join(path, 'msci_country_indices.csv'),
                         sep=';',
                         index_col=0,
                         header=0,
                         parse_dates=True)
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
        series_id = df.columns[0:n]
        X = df[series_id]
    except Exception as e:
        raise FileNotFoundError(f"Error loading MSCI country indices data: {e}")

    # Load MSCI World index return series
    try:
        y = pd.read_csv(f'{path}NDDLWI.csv',
                        sep=';',
                        index_col=0,
                        header=0,
                        parse_dates=True)
        y.index = pd.to_datetime(y.index, format='%d/%m/%Y')
    except Exception as e:
        raise FileNotFoundError(f"Error loading MSCI World index data: {e}")

    return {'return_series': X, 'bm_series': y}


