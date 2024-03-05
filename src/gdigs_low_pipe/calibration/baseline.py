
import numpy as np

from ..utils.core import rolling_window_slice 
from ..rfi import arpls


def baseline_arpls(x, y, lam, ratio=0.05, itermax=10):
    """
    """

    bl = arpls.ArPLS(y.compressed(), lam=lam, ratio=ratio, itermax=itermax)
    
    yblr = y.copy()
    yblr[~y.mask] = y.compressed() - bl

    bl_out = np.empty_like(yblr)
    bl_out[~y.mask] = bl
    bl_out[y.mask] = np.nan

    return yblr, bl_out


def baseline_poly(x, y, order):
    """
    """

    pfit = np.polyfit(x[~y.mask], y[~y.mask], order)
    bl = np.poly1d(pfit)(x)

    yblr = y.copy()
    yblr[~y.mask] = y.compressed() - bl[~y.mask]

    bl_out = np.empty_like(yblr)
    bl_out[~y.mask] = bl[~y.mask]
    bl_out[y.mask] = np.nan

    return yblr, bl_out


def baseline_loop(data, rms_window=32, method=baseline_arpls, **kwargs):
    """
    """

    new_data = np.empty_like(data)
    bl_arr = np.empty_like(data)
    rms_arr = np.empty_like(data)

    ntimes = data.shape[0]
    nchan = data.shape[1]

    slices = rolling_window_slice(rms_window, nchan-(rms_window-1), rms_window)
    x = np.linspace(-nchan//2, nchan//2, nchan)

    for i in range(ntimes):
        if np.all(np.isnan(data[i])):
            #print(i)
            continue
        y = np.ma.masked_invalid(data[i])
        # Compute baseline.
        try:
            new_data[i], bl_arr[i] = method(x, y, **kwargs)
        except TypeError:
            print(i)
            return -1
        # Compute rolling rms.
        rms_arr[i,16:-15] = np.roll(np.nanstd(new_data[i][slices], axis=1), rms_window//2)
        rms_arr[i,:16] = np.nan
        rms_arr[i,-15:] = np.nan

    out = {"baselined_data": new_data,
           "baseline": bl_arr,
           "rms": rms_arr
          }

    return out


def baseline(data, lam=1e7, rms_window=32):
    """
    """

    new_data = np.empty_like(data)
    bl_arr = np.empty_like(data)
    rms_arr = np.empty_like(data)

    ntimes = data.shape[0]
    nchan = data.shape[1]

    slices = rolling_window_slice(rms_window, nchan-(rms_window-1), rms_window)

    for i in range(ntimes):
        y = np.ma.masked_invalid(data[i])
        if np.all(np.isnan(y)):
            print(i)
            continue
        # Compute baseline.
        bl = arpls.ArPLS(y.compressed(), lam=lam)
        new_data[i] = y.copy()
        new_data[i][~y.mask] = y.compressed() - bl
        bl_arr[i][~y.mask] = bl
        bl_arr[i][y.mask] = np.nan
        # Compute rolling rms.
        rms_arr[i,16:-15] = np.roll(np.nanstd(new_data[i][slices], axis=1), rms_window//2)
        rms_arr[i,:16] = np.nan
        rms_arr[i,-15:] = np.nan

    out = {"baselined_data": new_data,
           "baseline": bl_arr,
           "rms": rms_arr
          }
    
    return out

