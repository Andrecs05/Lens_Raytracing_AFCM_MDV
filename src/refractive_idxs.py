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