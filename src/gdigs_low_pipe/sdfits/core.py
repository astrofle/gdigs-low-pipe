"""
Core functions for handling SDFITS files.
"""

import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.io import fits


def drop_data(table):
    """
    Copy the contents of a SDFITS 
    table without the data column.

    Parameters
    ----------
    table :
        The table to copy.

    Returns
    -------
    output :
        A copy of `table` without the DATA column.
    """

    names = table.names
    newdtype = []

    for name in names:
        if name in ["DATA"]:
            continue
        else:
            newdtype.append((name, table[name].dtype))

    output = np.empty(table.shape, dtype=newdtype)
    output = rfn.recursive_fill_fields(table, output)

    return output


def reshape_nchan(table, new_nchan, crval1=None, crpix1=None, cdelt1=None, 
                  ctype1=None, cunit1=None):
    """
    """

    ntime = len(table)
    #nodata_table = rfn.drop_fields(table.copy(), "DATA")
    nodata_table = drop_data(table)
    nodata_table_dt = nodata_table.dtype.descr
    data_dt = [("DATA", ">f4", (new_nchan,))]
    new_dt = np.dtype(nodata_table_dt[:6] + data_dt + nodata_table_dt[6:])
    new_table = np.empty(ntime, dtype=new_dt)

    for n in nodata_table.dtype.names:
        new_table[n] = nodata_table[n]
    if crval1 is not None:
        new_table["CRVAL1"] = crval1
    if crpix1 is not None:
        new_table["CRPIX1"] = crpix1
    if cdelt1 is not None:
        new_table["CDELT1"] = cdelt1
    if ctype1 is not None:
        new_table["CTYPE1"] = ctype1
    if cunit1 is not None:
        print("CUNIT1 not implemented!")

    return new_table


def copy_new_table(sdf, table, output, overwrite=False):
    """
    Copy the primary header from an SDFITS,
    create a new table HDU from `table` and
    write it to output.
    """

    hdu0 = sdf._hdu[0].copy()
    outhdu = fits.HDUList(hdu0)
    hdu = fits.BinTableHDU.from_columns(table)
    hdu.header["EXTNAME"] = "SINGLE DISH"
    outhdu.append(hdu)
    outhdu.writeto(output, overwrite=overwrite)


def write_new_table(phdu, table, output, overwrite=False):
    """
    Copy the primary header from an SDFITS,
    create a new table HDU from `table` and
    write it to output.
    """

    #hdu0 = sdf._hdu[0].copy()
    outhdu = fits.HDUList(phdu)
    #hdu = fits.BinTableHDU.from_columns(table)
    hdu = fits.BinTableHDU(data=table)
    hdu.header["EXTNAME"] = "SINGLE DISH"
    outhdu.append(hdu)
    outhdu.writeto(output, overwrite=overwrite)
