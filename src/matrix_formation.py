import numpy as np

def doublet_matrix(n1, n2, n3, R1, R2, R3, d1, d2):
    """
    Calculate the transfer matrix for a doublet lens system.

    Args:
        n1 : float - Refractive index of the medium before the first lens
        n2 : float - Refractive index of the first lens
        n3 : float - Refractive index of the second lens
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        R3 : float - Radius of curvature of the third surface (positive if center to the right)
        d1 : float - Width of the first lens
        d2 : float - Width of the second lens
    
    Returns:
        MT : numpy.ndarray - 2x2 transfer matrix for the doublet lens system

    """
    M1 = np.array([[1, 0], [(n1 - n2) / (R1 * n2), n1 / n2]])
    M2 = translation_matrix(d1, n2)
    M3 = np.array([[1, 0], [(n2 - n3) / (R2 * n3), n2 / n3]])
    M4 = translation_matrix(d2, n3)
    M5 = np.array([[1, 0], [(n3 - n1) / (R3 * n1), n3 / n1]])

    MT = M5 @ M4 @ M3 @ M2 @ M1

    f1 = 1 / ((n2 - n1) * (1 / R1 - 1 / R2)) 
    f2 = 1 / ((n3 - n1) * (1 / R2 - 1 / R3))  # Focal length of the lens (in mm)
    f = 1 / ((1 / f1) + (1 / f2))  # Focal length of the lens (in mm)

    return MT, f

def translation_matrix(d, n):
    """
    Calculate the translation matrix for a distance z.

    Args:
        z : float - Distance to translate

    Returns:
        M : numpy.ndarray - 2x2 translation matrix
    """
    M = np.array([[1, d/n],
                  [0, 1]])
    return M

def thick_lens_transfer_matrix(R1, R2, d, n):
    """
    Calculate the transfer matrix for a thick lens.

    Args:
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        d : float - Thickness of the lens
        n : float - Refractive index of the lens material
    
    Returns:
        M : numpy.ndarray - 2x2 transfer matrix for the thick lens
    """
    M1 = np.array([[1, 0], [(1-n)/R1, 1]])
    M_prop = np.array([[1, d/n], [0, 1]])
    M2 = np.array([[1, 0], [(n-1)/R2, 1]])
    M = M2 @ M_prop @ M1
    return M

def thin_lens_transfer_matrix(f):
    """
    Calculate the transfer matrix for a thin lens.

    Args:
        f : float - Focal length of the lens
    Returns:
        M : numpy.ndarray - 2x2 transfer matrix for the thin lens
    """
    M = np.array([[1, 0], [-1/f, 1]])
    return M

def reflection_matrix(n1, n2, R):
    """
    Calculate the reflection matrix for a spherical surface.
    Args:
        n1 : float - Refractive index of the medium before the surface
        n2 : float - Refractive index of the medium after the surface
        R : float - Radius of curvature of the surface (positive if center to the right)
    Returns:
        M : numpy.ndarray - 2x2 reflection matrix for the spherical surface
    """
    M = np.array([[1, 0], [(n1 - n2) / (R * n2), n1 / n2]])
    return M

def interface_matrix(n1, n2):
    """
    Calculate the interface matrix for a planar interface between two media.
    Args:
        n1 : float - Refractive index of the medium before the interface
        n2 : float - Refractive index of the medium after the interface
    Returns:
        M : numpy.ndarray - 2x2 interface matrix for the planar interface
    """
    M = np.array([[1, 0], [0, n1 / n2]])
    return M
