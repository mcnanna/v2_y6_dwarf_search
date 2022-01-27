#!/usr/bin/env python
import numpy as np

def magToFlux(mag):
    """
    Convert from an AB magnitude to a flux (Jy)
    """
    return 3631. * 10**(-0.4 * mag)

def fluxToMag(flux):
    """
    Convert from flux (Jy) to AB magnitude
    """
    with np.errstate(invalid='ignore'):
        mag =-2.5 * np.log10(flux / 3631.)
    return mag

def getFluxError(mag, mag_error):
    return magToFlux(mag) * mag_error / 1.0857362


a, b = -2.51758, 4.86721 # From abs_mag.py
def mass_to_mag(stellar_mass):
    return a*np.log10(stellar_mass) + b
def mag_to_mass(m_v):
    return 10**((m_v-b)/a)

def ra_dec(x, y, z):
        phi = np.arctan2(y, x)
        phi = np.array([(p if p>0 else p+2*np.pi) for p in phi])
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        ra = np.degrees(phi)
        dec = 90-np.degrees(theta)

        return ra, dec
