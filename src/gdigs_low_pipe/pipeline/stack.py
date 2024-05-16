"""
"""

import numpy as np
import pandas as pd

from astropy.io import fits

from gdigs_low_pipe import sdfits


def stack(files, line_list, output, 
          poly_order=1, left=400, right=400,
          verbose=True):
    """

    Parameters
    ----------
    files : list

    """

    df = pd.read_csv(line_list)

    #print(f"Will stack {df['use'].sum()} lines out of {len(df)}.")

    ntime = fits.getval(files[0], "NAXIS2", ext=1)
    nchan = int(fits.getval(files[0], "TDIM7", ext=1)[1:-1])

    x_axis = np.arange(0,nchan,1) - nchan/2

    stack = np.zeros((ntime, nchan), dtype='d')
    weights = np.zeros((ntime, nchan), dtype='d')
    count = np.zeros((ntime, nchan), dtype='l')
    level = np.zeros((ntime, nchan), dtype='d')
    tsys = np.zeros((ntime), dtype='d')
    exposure = np.zeros((ntime), dtype='d')

    header = False

    # Loop over files process and add them to the stack.
    for i,f in enumerate(files):
        f = str(f)
        ifn = int(f.split("/")[-1].split("_")[1])
        pln = int(f.split("/")[-1].split("_")[3])
        pqn = int(f.split("/")[-1].split("_")[6].split(".")[0])

        use = df[(df["ifnum"] == ifn) & (df["plnum"] == pln) & (df["n"] == pqn)]["use"].values[0]

        if use:

            hdu = fits.open(f, readonly=True)
            table = hdu[1].data
            data = table["DATA"]
            mask = ~np.isfinite(table["DATA"])

            if verbose:
                print(f"Non finite values in data: {np.sum(mask)}/{mask.size} -- {np.sum(mask)/mask.size*100.}")

            for j,row in enumerate(data):
                if mask[j].sum() > nchan/2:
                    continue
                x = x_axis[~mask[j]]
                y = row[~mask[j]]
                pfit = np.polyfit(x, y, poly_order)
                pval = np.poly1d(pfit)(x_axis)
                data[j] -= pval

            radiom = table["EXPOSURE"]/table["TSYS"]**2
            weight = np.empty_like(data)
            wleft = 1./np.nanvar(data[:,:left], axis=1)
            wright = 1./np.nanvar(data[:,-right:], axis=1)
            weight[:,:] = np.min(np.c_[wleft, wright], axis=1)[:,np.newaxis]
            weight[mask] = 0.0

            if verbose:
                print(f"Non finite values in weights: {np.sum(~np.isfinite(weight))}/{weight.size}")

            count += (~mask).astype('l')
            stack[~mask] += (data*weight)[~mask]
            level[~mask] += (pqn*weight)[~mask]
            weights += weight
            tsys += table["TSYS"]*weight[:,0]
            exposure += table["EXPOSURE"]*weight[:,0]

            if not header:
                # Save the spectral axis and primary HDU.
                cdelt1      = table["CDELT1"]
                crval1      = table["CRVAL1"]
                crpix1      = table["CRPIX1"]
                phdu        = hdu[0].copy()
                stack_table = table.copy()

                header = True

    # Combine.
    stack /= weights
    level /= weights
    tsys /= weights.mean(axis=1)
    exposure /= weights.mean(axis=1)

    stack_table["DATA"] = stack
    stack_table["EXPOSURE"] = exposure
    stack_table["TSYS"] = tsys
    stack_table["CDELT1"] = cdelt1
    stack_table["CRVAL1"] = crval1
    stack_table["CRPIX1"] = crpix1

    output_ = f"{output}.fits"
    print(f"Writting: {output_}")
    sdfits.write_new_table(phdu, stack_table, output_, overwrite=True)
    hdu = fits.open(output_)
    stable = hdu[1].data

    stack_table["DATA"] = count
    output_ = f"{output}_cov.fits"
    print(f"Writting: {output_}")
    sdfits.write_new_table(phdu, stack_table, output_, overwrite=True)
    hdu = fits.open(output_)
    ctable = hdu[1].data

    stack_table["DATA"] = level
    output_ = f"{output}_pqn.fits"
    print(f"Writting: {output_}")
    sdfits.write_new_table(phdu, stack_table, output_, overwrite=True)
    hdu = fits.open(output_)
    ntable = hdu[1].data
