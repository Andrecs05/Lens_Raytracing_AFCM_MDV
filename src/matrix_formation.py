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
    """
    M1 = refraction_matrix(n1, n2, R1)
    M2 = translation_matrix(d1, n2)
    M3 = refraction_matrix(n2, n3, R2)
    M4 = translation_matrix(d2, n3)
    M5 = refraction_matrix(n3, n1, R3)

    MT = M5 @ M4 @ M3 @ M2 @ M1

    C = MT[1,0]
    f = -1/C 

    return MT, f

def triplet_matrix(n1, n2, n3, n4, R1, R2, R3, R4, d1, d2, d3):
    """
    Calculate the transfer matrix for a triplet lens system.

    Args:
        n1 : float - Refractive index of the medium before the first lens
        n2 : float - Refractive index of the first lens
        n3 : float - Refractive index of the second lens
        n4 : float - Refractive index of the third lens
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        R3 : float - Radius of curvature of the third surface (positive if center to the right)
        R4 : float - Radius of curvature of the fourth surface (positive if center to the right)
        d1 : float - Width of the first lens
        d2 : float - Width of the second lens
        d3 : float - Width of the third lens
    """
    M1 = refraction_matrix(n1, n2, R1)
    M2 = translation_matrix(d1, n2)
    M3 = refraction_matrix(n2, n3, R2)
    M4 = translation_matrix(d2, n3)
    M5 = refraction_matrix(n3, n4, R3)
    M6 = translation_matrix(d3, n4)
    M7 = refraction_matrix(n4, n1, R4)

    MT = M7 @ M6 @ M5 @ M4 @ M3 @ M2 @ M1

    C = MT[1,0]
    f = -1/C 

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
    M1 = refraction_matrix(1, n, R1)
    M_prop = translation_matrix(d, n)
    M2 = refraction_matrix(n, 1, R2)
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

def reflection_matrix(R):
    """
    Reflection at a spherical mirror (paraxial, y-theta formalism)

    R : radius of curvature (positive if center to the right)
    """
    return np.array([
        [1, 0],
        [-2/R, 1]
    ])

def refraction_matrix(n_in, n_out, R):
    """
    Refraction at spherical surface (y, theta formalism)
    """
    return np.array([
        [1, 0],
        [(n_in - n_out) / R, 1]
    ], dtype=float)

