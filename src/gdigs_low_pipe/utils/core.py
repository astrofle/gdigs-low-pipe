"""
"""

from pathlib import Path

import numpy as np
import bottleneck as bn


def find_blanks(data, verbose=True):
    """
    """

    blank_mask = np.all(np.isnan(data), axis=1)
    if verbose:
        print(f"Found {blank_mask.sum()} blanked integrations.")

    return blank_mask


def get_vegas_sdfits_files(path):
    """
    Return the VEGAS SDFITS files in path.

    Parameters
    ----------
    path : str
        The path where to look for VEGAS SDFITS files.

    Returns
    -------
    files : list
        A list of VEGAS SDFITS files in `path`.
    """

    files = Path(path).rglob("*.vegas.*.fits")

    return files


def rolling(x, func=bn.move_median, win=32):
    """
    Parameters
    ----------
    x : array
        Array to apply rolling window function to.
    func : function
        Function to apply.
    win : int
        Window width in channels.
    """

    return np.roll(func(x, win), -win//2)


def rolling_window_slice(start_index, max_index, window_size):
    """
    Create indices to perform rolling window operations
    on an array.
    Taken from:
    https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    
    Parameters
    ----------
    start_index : int
        Ignore indices before this value at the start.
    max_index : int
        Maximum index to include in the slices.
    window_size : int
        Number of elements to include in each slice.

    Returns
    -------
    windows : array
        Array with the indices of each window.
    """

    start = start_index - window_size

    windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(max_index), 0).T
    )

    return windows

