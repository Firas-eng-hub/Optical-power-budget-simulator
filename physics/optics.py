import numpy as np

# Sellmeier coefficients for pure silica (Malitson 1965)
_A = [0.6961663, 0.4079426, 0.8974794]
_L = [0.0684043, 0.1162414, 9.896161]  # resonance wavelengths in µm


def sellmeier_index(wavelength_um):
    """
    Refractive index of silica via the Sellmeier equation.
    wavelength_um: wavelength in micrometres (scalar or numpy array)
    Returns n(lambda).
    """
    lam = np.asarray(wavelength_um, dtype=float)
    n2 = 1.0
    for Ai, Li in zip(_A, _L):
        n2 = n2 + Ai * lam**2 / (lam**2 - Li**2)
    return np.sqrt(n2)


def group_index(wavelength_um: float) -> float:
    """
    Group index ng = n - lambda * dn/dlambda, computed by finite difference.
    """
    delta = 1e-5  # µm
    n_plus = float(sellmeier_index(wavelength_um + delta))
    n_minus = float(sellmeier_index(wavelength_um - delta))
    dn_dlam = (n_plus - n_minus) / (2 * delta)
    n = float(sellmeier_index(wavelength_um))
    return n - wavelength_um * dn_dlam


def numerical_aperture(n1: float, n2: float) -> float:
    """NA = sqrt(n1^2 - n2^2)"""
    val = n1**2 - n2**2
    return float(np.sqrt(max(val, 0.0)))


def relative_index_diff(n1: float, n2: float) -> float:
    """Delta = (n1 - n2) / n1"""
    if n1 == 0:
        return 0.0
    return (n1 - n2) / n1
