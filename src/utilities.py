import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

def gal_telescope_transfer_matrix(n, R1o, R2o, do, fo, R1e, R2e, de, fe):
    """
    Calculate the transfer matrix for a Galilean telescope.

    Args:
        n1 : float - Refractive index of the medium before the first lens
        n2 : float - Refractive index of the medium after the first lens
        R1o : float - Radius of curvature of the first lens (positive if center to the right)
        R2o : float - Radius of curvature of the second lens (positive if center to the right)
        do : float - Distance between the two lenses
        fo : float - Focal length of the first lens
        R1e : float - Radius of curvature of the eyepiece lens (positive if center to the right)
        R2e : float - Radius of curvature of the eyepiece lens (positive if center to the right)
        de : float - Distance between the eyepiece and image plane
        fe : float - Focal length of the eyepiece
    
    Returns:
        M : numpy.ndarray - 2x2 transfer matrix for the Galilean telescope
    """

    Mo = thick_lens_transfer_matrix(R1o, R2o, do, n)
    Me = thick_lens_transfer_matrix(R1e, R2e, de, n)
    D_oe = fo + fe
    MD = translation_matrix(D_oe, 1)
    MT = Me @ MD @ Mo
    return MT

def transform_coords(x, y, theta, M):
    """
    Transform a pixel using the transfer matrix.

    Args:
        center_y : float - Y-coordinate of the center
        center_x : float - X-coordinate of the center
        pixel : numpy.ndarray - 2D array representing the pixel (x, y)
        r : float - Radial distance
        theta : float - Angle in radians
        M : numpy.ndarray - 2x2 transfer matrix

    Returns:
        transformed_pixel : numpy.ndarray - Transformed pixel (x', y')
        final_state : numpy.ndarray - Final state after transformation
    """
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)
    initial_state = np.array([r, theta])

    final_state = M @ initial_state
    r_f = final_state[0]
    x_f = r_f * np.cos(alpha)
    y_f = r_f * np.sin(alpha)

    return x_f, y_f, final_state

def get_doublet_parameters(img_array, n1, n1_o, n2_o, n1_e, n2_e, R1_o, R2_o, R3_o, d1_o, d2_o,
                           R1_e, R2_e, R3_e, d1_e, d2_e):
    """
    Calculate the parameters for a doublet lens system.

    Args:
        img_array : numpy.ndarray - Input image array
        n1 : float - Refractive index of the medium before the first lens
        n1_o : float - Refractive index of the first lens
        n2_o : float - Refractive index of the second lens
        n1_e : float - Refractive index of the eyepiece lens
        n2_e : float - Refractive index of the medium after the eyepiece lens
        R1_o, R2_o, R3_o : float - Radii of curvature for the objective lens
        d1_o, d2_o : float - Widths for the objective lens
        R1_e, R2_e, R3_e : float - Radii of curvature for the eyepiece lens
        d1_e, d2_e : float - Widths for the eyepiece lens
        f_o : float - Focal length of the objective lens (in mm)
        f_e : float - Focal length of the eyepiece lens (in mm)

    Returns:
        fov_i : float - Initial field of view
        f_o : float - Focal length of the objective lens (in mm)
        f_e : float - Focal length of the eyepiece lens (in mm)
        i_angular_pixel_size : float - Initial angular pixel size in the image plane
        i_lateral_pixel_size : float - Initial lateral pixel size in the image plane
        output_img_size : tuple - Size of the output image (height, width)
        M_T : numpy.ndarray - Transfer matrix for the doublet lens system
        M : float - Magnification factor
    """

    H, W = img_array.shape 

    M_o, f_o = doublet_matrix(n1, n1_o, n2_o, R1_o, R2_o, R3_o, d1_o, d2_o)
    M_e, f_e = doublet_matrix(n1_e, n2_e, n1, R1_e, R2_e, R3_e, d1_e, d2_e)
    d = f_o + f_e
    M_T = M_e @ translation_matrix(d, 1) @ M_o

    M = (-f_o / f_e)

    fov_i = 25.4/f_o

    max_img_size = max(img_array.shape)

    i_angular_pixel_size = fov_i / max_img_size

    i_lateral_pixel_size = np.tan(i_angular_pixel_size) * f_o

    output_img_size = (int(np.ceil(H * abs(M))), int(np.ceil(W * abs(M))))

    return fov_i, f_o, f_e, i_angular_pixel_size, i_lateral_pixel_size, output_img_size, M_T, M

def get_parameters(img_array, n1, n2, R1_o, R2_o, d_o, R1_e, R2_e, d_e):
    """
    Calculate the parameters for a telescope system.

    Args:
        img_array : numpy.ndarray - Input image array
        n1 : float - Refractive index of the medium before the first lens
        n2 : float - Refractive index of the medium after the first lens
        R1_o, R2_o : float - Radii of curvature for the objective lens
        d_o : float - Distance between the two lenses
        R1_e, R2_e : float - Radii of curvature for the eyepiece lens
        d_e : float - Distance between the eyepiece and image plane
    
    Returns:
        fov_i : float - Initial field of view
        f_o : float - Focal length of the objective lens (in mm)
        f_e : float - Focal length of the eyepiece lens (in mm)
        i_angular_pixel_size : float - Initial angular pixel size in the image plane
        i_lateral_pixel_size : float - Initial lateral pixel size in the image plane
        output_img_size : tuple - Size of the output image (height, width)
        M_T : numpy.ndarray - Transfer matrix for the telescope system
        M : float - Magnification factor
    """

    f_o = 1 / ((n2 - n1) * (1 / R1_o - 1 / R2_o))  
    f_e = 1 / ((n2 - n1) * (1 / R1_e - 1 / R2_e)) 

    H, W = img_array.shape  

    M_T = gal_telescope_transfer_matrix(n2, R1_o, R2_o, d_o, f_o, R1_e, R2_e, d_e, f_e)

    M = (-f_o / f_e)

    fov_i = 25.4/f_o

    max_img_size = max(img_array.shape)
    i_angular_pixel_size = fov_i / max_img_size
    i_lateral_pixel_size = np.tan(i_angular_pixel_size) * f_o
    output_img_size = (int(np.ceil(H * abs(M))), int(np.ceil(W * abs(M))))

    return fov_i, f_o, f_e, i_angular_pixel_size, i_lateral_pixel_size, output_img_size, M_T, M

def telescope(img_array, fov_i, f_o, f_e, i_angular_pixel_size, i_lateral_pixel_size, final_img_size, M_T, M):
    """
    Apply the telescope transformation to the input image array.

    Args:
        img_array : numpy.ndarray - Input image array
        fov_i : float - Initial field of view
        f_o : float - Focal length of the objective lens (in mm)
        f_e : float - Focal length of the eyepiece lens (in mm)
        i_angular_pixel_size : float - Initial angular pixel size in the image plane
        i_lateral_pixel_size : float - Initial lateral pixel size in the image plane
        final_img_size : tuple - Size of the output image (height, width)
        M_T : numpy.ndarray - Transfer matrix for the telescope system
        M : float - Magnification factor

    """

    H, W = img_array.shape 

    fov_o = fov_i * M

    max_output_img_size = max(final_img_size)
    f_angular_pixel_size = fov_o / max_output_img_size
    f_lateral_pixel_size = np.tan(f_angular_pixel_size) * abs(f_e)

    final_img = np.zeros(final_img_size, dtype=np.uint8)
    H_out, W_out = final_img_size

    x = np.arange(W) - (W - 1) / 2  
    y = np.arange(H) - (H - 1) / 2 
    xx, yy = np.meshgrid(x, y, indexing='xy')  
    x_world = xx * i_lateral_pixel_size
    y_world = yy * i_lateral_pixel_size
    theta = np.sqrt((xx * i_angular_pixel_size)**2 + (yy * i_angular_pixel_size)**2)

    for y in range(H):
        for x in range(W):
            x_i = x_world[y, x]
            y_i = y_world[y, x]
            theta_i = theta[y, x]

            x_f, y_f, final_state = transform_coords(x_i, y_i, theta_i, M_T)

            x_p = (x_f / f_lateral_pixel_size + (W_out - 1) / 2).astype(int)
            y_p = (y_f / f_lateral_pixel_size + (H_out - 1) / 2).astype(int)

            if (0 <= x_p < final_img_size[1] and 0 <= y_p < final_img_size[0]):
                final_img[y_p, x_p] = img_array[y, x]

    return final_img

