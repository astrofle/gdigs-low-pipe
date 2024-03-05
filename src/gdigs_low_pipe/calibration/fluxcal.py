"""
Flux scale calibration functions.
"""

def quad_cal(freq, sig_on, sig_off, ref_on, ref_off, ta_src):
    """
    Computes the non-linear gain, system temperature and noise diode temperature.
    
    Parameters
    ----------
    freq : array
        Frequency.
    sig_on : array
        Spectra on source with the noise diode on.
    sig_off : array
        
    """

    Q = -(sig_on - ref_on - sig_off + ref_off)/(sig_on**2. - ref_on**2. - sig_off**2. + ref_off**2.)
    H = ta_src * (1./(sig_on - ref_on) - 1./(sig_off - ref_off))/((sig_on + ref_on) - (sig_off + ref_off))
    G = H/Q
    Tsys = G*ref_off + H*ref_off**2.
    Tcal = G*ref_on + H*ref_on**2. - Tsys

    out = {"G": G, "H": H, "Tsys": Tsys, "Tcal": Tcal}

    return out


def linear_cal(freq, sig_on, sig_off, ref_on, ref_off, ta_src):
    """
    """

    G = ta_src/(sig_off - ref_off)
    Tcal = ta_src * (ref_on - ref_off) / (sig_off - ref_off)
    Tsys = ref_off*G

    out = {"G": G, "Tsys": Tsys, "Tcal": Tcal}

    return out
