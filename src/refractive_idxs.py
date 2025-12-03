import numpy as np

def refractive_index_NBK7(lambda_um):
    """
    Calculates the refractive index n of the material NBK-7 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945

    C1 = 0.00600069867
    C2 = 0.0200179144
    C3 = 103.560653

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )
    
    return np.sqrt(n_squared)

def refractive_index_NBAF10(lambda_um):
    """
    Calculates the refractive index n of the material NBAF10 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.5851495, 0.00926681282
    B2, C2 = 0.143559385, 0.0424489805
    B3, C3 = 1.08521269, 105.613573
    
    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_NSF6HT(lambda_um):
    """
    Calculates the refractive index n of the material NSF6HT for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.77931763, 0.0133714182
    B2, C2 = 0.338149866, 0.0617533621
    B3, C3 = 2.08734474, 174.01759

    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_NSF2(lambda_um):
    """
    Calculates the refractive index n of the material NSF2 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.47343127, 0.0109019098
    B2, C2 = 0.163681849, 0.0585683687
    B3, C3 = 1.36920899, 127.404933

    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_BaF2(lambda_um):
    """
    Calculates the refractive index n of BaF2 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of BaF2.
    """
    A = 0.33973
    B1, C1 = 0.81070, 0.10065**2
    B2, C2 = 0.19652, 29.87**2
    B3, C3 = 4.52469, 53.82**2

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_CaF2(lambda_um):
    """
    Calculates the refractive index n of CaF2 using the Sellmeier equation.

    Valid for wavelengths from 0.15 to 12.0 µm.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of CaF2.
    """
    A = 0.33973
    B1, C1 = 0.69913, 0.09374**2
    B2, C2 = 0.11994, 21.18**2
    B3, C3 = 4.35181, 38.46**2

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_Ge(lambda_um):
    """
    Calculates the refractive index n of Germanium (Ge) using the Sellmeier equation.

    Valid for wavelengths from 2.5 to 12.0 µm.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of Ge.
    """
    A = 8.28156
    B1, C1 = 6.72888, 0.44105
    B2, C2 = 0.21307, 3870.1

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2)
    )

    return np.sqrt(n_squared)

def refractive_index_MgF2(lambda_um):
    """
    Calculates the refractive index n of MgF2 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of MgF2.
    """
    B1, C1 = 0.48755108, 0.04338408**2
    B2, C2 = 0.39875031, 0.09461442**2
    B3, C3 = 2.3120353, 23.793604**2

    lambda_sq = lambda_um**2

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_SF11(lambda_um):
    """
    Calculates the refractive index n of N-SF11 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-SF11.
    """
    lambda_sq = lambda_um**2

    # Sellmeier coefficients (from provided equation)
    B1, C1 = 1.73759695, 0.013188707
    B2, C2 = 0.313747346, 0.0623068142
    B3, C3 = 1.89878101, 155.23629

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)