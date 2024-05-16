"""
Equatorial to Galactic coordinate conversion.
"""

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord


def eq2gal_table(table):
    """
    Transform the coordinates in `table` from Equatorial to Galactic.

    Parameters
    ----------
    table : `~astropy.FITS_rec`
        SDFITS table with the data and coordinates.

    Returns
    -------
    table : `~astropy.FITS_rec`
        SDFITS table with all Equatorial coordinates
        changed to Galactic.
    """

    mask = (table["CTYPE2"] == "RA  ") & (table["CTYPE3"] == "DEC ")
    coo = SkyCoord(table["CRVAL2"][mask]*u.deg, table["CRVAL3"][mask]*u.deg, frame="fk5")

    glon = coo.galactic.l.deg
    glat = coo.galactic.b.deg

    table["CTYPE2"][mask] = "GLON"
    table["CRVAL2"][mask] = glon

    table["CTYPE3"][mask] = "GLAT"
    table["CRVAL3"][mask] = glat

    return table


def eq2gal(filename, mode="update"):
    """
    """

    hdu = fits.open(filename, mode=mode)
    table = hdu[1].data
    table = eq2gal_table(table)
    hdu[1].data = table
    if mode == "update":
        hdu.flush()
    else:
        raise(TypeError, "Not implemented")
