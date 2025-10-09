import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from src.matrix_formation import *


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

