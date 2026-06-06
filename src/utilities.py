import numpy as np
from src.refractive_idxs import *
from src.matrix_formation import *



def image_to_rays_0_angle(image_array, pixel_size, channel=0):
    '''
    Convert a 2D image array into a list of rays for ray tracing.
    
    Parameters:
    image_array : 2D numpy array - The input image where each pixel represents an intensity value.
    pixel_size : float - The physical size of each pixel in the same units as the ray tracing system.
    channel : int - The channel index to extract intensity values from (default is 0).
    
    Returns:
    rays : list of tuples - A list of rays represented as tuples (x, y, angle, intensity) where:
        x : float - The x-coordinate of the ray's origin
        y : float - The y-coordinate of the ray's origin
        angle : float - The angle of the ray with respect to the optical axis (in radians)
        intensity : float - The intensity of the ray based on the pixel value
    '''
    rays = []
    height, width, channels = image_array.shape
    for i in range(height):
        for j in range(width):
            intensity = image_array[i, j, channel]  # Assuming the first channel represents intensity
            if intensity > 0:  # Only consider pixels with non-zero intensity
                x = j * pixel_size  # Calculate x-coordinate based on pixel index and size
                y = i * pixel_size  # Calculate y-coordinate based on pixel index and size
                angle = 0  # Initial angle of the ray (can be modified later based on the system)
                rays.append((x, y, angle, intensity))  # Append the ray with its intensity for further processing
    # Center the rays around the optical axis by adjusting their x and y coordinates
    rays = [(x - (width * pixel_size) / 2, y - (height * pixel_size) / 2, angle, intensity) for (x, y, angle, intensity) in rays]
    return rays

def image_infinity_to_rays(image_array, channel=0):
    '''
    Convert a 2D image array into a list of rays for ray tracing, assuming the image is at infinity.
    
    Parameters:
    image_array : 2D numpy array - The input image where each pixel represents an intensity value.
    channel : int - The channel index to extract intensity values from (default is 0).
    
    Returns:
    rays : list of tuples - A list of rays represented as tuples (x, y, angle, intensity) where:
        x : float - The x-coordinate of the ray's origin
        y : float - The y-coordinate of the ray's origin
        angle : float - The angle of the ray with respect to the optical axis (in radians)
        intensity : float - The intensity of the ray based on the pixel value
    '''
    rays = []
    height, width, channels = image_array.shape
    cx = width / 2  # Calculate the center x-coordinate of the image
    cy = height / 2  # Calculate the center y-coordinate of the image
    for i in range(height):
        for j in range(width):
            intensity = image_array[i, j, channel]  # Assuming the first channel represents intensity
            if intensity > 0:  # Only consider pixels with non-zero intensity
                anglex = (j - cx) / cx
                angley = (i - cy) / cy
                rays.append((anglex, angley, intensity))  # Append the ray with its intensity for further processing
    return rays, cx, cy

def rays_to_image(rays, pixel_size):
    '''
    Convert a list of rays back into a 2D image array.
    
    Parameters:
    rays : list of tuples - A list of rays represented as tuples (x, y, angle, intensity) where:
        x : float - The x-coordinate of the ray's origin
        y : float - The y-coordinate of the ray's origin
        angle : float - The angle of the ray with respect to the optical axis (in radians)
        intensity : float - The intensity of the ray
    pixel_size : float - The physical size of each pixel in the same units as the ray tracing system.
    
    Returns:
    image_array : 2D numpy array - The resulting image array where each pixel value is the sum of intensities from rays that fall into that pixel.
    '''
    # Determine the size of the image based on the maximum x and y coordinates of the rays
    max_x = max(ray[0] for ray in rays)
    max_y = max(ray[1] for ray in rays)
    width = int(np.ceil(max_x * 2/ pixel_size)) + 1  # Calculate the width of the image array
    height = int(np.ceil(max_y * 2/ pixel_size)) + 1  # Calculate the height of the image array
    image_array = np.zeros((height, width), dtype=np.float32)  # Initialize the image array with zeros
    for ray in rays:
        x_centered, y_centered, angle, intensity = ray
        x = x_centered + (width * pixel_size) / 2  # Adjust x-coordinate back to image space
        y = y_centered + (height * pixel_size) / 2  # Adjust y-coordinate back to image space
        j = int(x / pixel_size)  # Calculate the pixel index for x-coordinate
        i = int(y / pixel_size)  # Calculate the pixel index for y-coordinate
        if 0 <= i < height and 0 <= j < width:  # Ensure the indices are within the bounds of the image array
            image_array[i, j] += intensity  # Add the intensity of the ray to the corresponding pixel
    # Normalize the image array to the range [0, 255] for visualization purposes
    image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
    return image_array

def rays_to_image_infinity(rays, cx, cy):
    '''
    Convert a list of rays back into a 2D image array, assuming the rays are coming from infinity.
    
    Parameters:
    rays : list of tuples - A list of rays represented as tuples (anglex, angley, intensity) where:
        anglex : float - The angle of the ray in the x-direction with respect to the optical axis (in radians)
        angley : float - The angle of the ray in the y-direction with respect to the optical axis (in radians)
        intensity : float - The intensity of the ray
    pixel_size : float - The physical size of each pixel in the same units as the ray tracing system.
    
    Returns:
    image_array : 2D numpy array - The resulting image array where each pixel value is the sum of intensities from rays that fall into that pixel.
    '''
    max_ux = max(abs(ray[0]) for ray in rays)
    max_uy = max(abs(ray[1]) for ray in rays)
    j_max = (int(np.ceil(max_ux * cx + cx)) + 1) * 2  # Calculate the maximum pixel index for x-direction
    i_max = (int(np.ceil(max_uy * cy + cy)) + 1) * 2  # Calculate the maximum pixel index for y-direction
    image_array = np.zeros((i_max, j_max), dtype=np.float32)  # Initialize the image array with zeros
    for ray in rays:
        ux_out, uy_out, intensity = ray
        x = int((ux_out + 1) * cx) + j_max // 2  # Convert angle back to pixel index for x-coordinate
        y = int((uy_out + 1) * cy) + i_max // 2  # Convert angle back to pixel index for y-coordinate
        if 0 <= y < image_array.shape[0] and 0 <= x < image_array.shape[1]:  # Ensure the indices are within the bounds of the image array
            image_array[y, x] += intensity  # Add the intensity of the ray to the corresponding pixel
    # Cut empty rows and columns from the image array to focus on the area where rays are present
    non_zero_rows = np.where(np.any(image_array > 0, axis=1))[0]
    non_zero_cols = np.where(np.any(image_array > 0, axis=0))[0]
    if non_zero_rows.size > 0 and non_zero_cols.size > 0:
        image_array = image_array[non_zero_rows[0]:non_zero_rows[-1] + 1, non_zero_cols[0]:non_zero_cols[-1] + 1]
    image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)  # Normalize the image array to the range [0, 255] for visualization purposes
    return image_array

def pad_to_shape(img, target_h, target_w):
    h, w = img.shape
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=0)

def update_elements_color(system, element, material=None):
    type = element.__class__.__name__
    if type == 'ThickLens':
                nR = refractive_index(0.6563, material[0])  # Refractive index for red light (656.3 nm)
                nG = refractive_index(0.5876, material[0])  # Refractive index for green light (587.6 nm)
                nB = refractive_index(0.4861, material[0])  # Refractive index for blue light (486.1 nm)
                element_MR = thick_lens_transfer_matrix(element.R1, element.R2, element.d, nR)  # Transfer matrix for red light
                element_MG = thick_lens_transfer_matrix(element.R1, element.R2, element.d, nG)  # Transfer matrix for green light
                element_MB = thick_lens_transfer_matrix(element.R1, element.R2, element.d, nB)  # Transfer matrix for blue light
                system.MR = element_MR @ system.MR  # Update the system matrix for red light
                system.MG = element_MG @ system.MG  # Update the system matrix for green light
                system.MB = element_MB @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    elif type == 'Mirror':
                system.MR = element.matrix() @ system.MR  # Update the system matrix for red light
                system.MG = element.matrix() @ system.MG  # Update the system matrix for green light
                system.MB = element.matrix() @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    elif type == 'Doublet':
                nR1 = refractive_index(0.6563, material[0])  # Refractive index for red light (656.3 nm) for the first lens
                nR2 = refractive_index(0.6563, material[1])  # Refractive index for red light (656.3 nm) for the second lens
                nG1 = refractive_index(0.5876, material[0])  # Refractive index for green light (587.6 nm) for the first lens
                nG2 = refractive_index(0.5876, material[1])  # Refractive index for green light (587.6 nm) for the second lens
                nB1 = refractive_index(0.4861, material[0])  # Refractive index for blue light (486.1 nm) for the first lens
                nB2 = refractive_index(0.4861, material[1])  # Refractive index for blue light (486.1 nm) for the second lens
                element_MR = doublet_matrix(1, nR1, nR2, element.R1, element.R2, element.R3, element.d1, element.d2)  # Transfer matrix for red light
                element_MG = doublet_matrix(1, nG1, nG2, element.R1, element.R2, element.R3, element.d1, element.d2)  # Transfer matrix for green light
                element_MB = doublet_matrix(1, nB1, nB2, element.R1, element.R2, element.R3, element.d1, element.d2)  # Transfer matrix for blue light
                system.MR = element_MR @ system.MR  # Update the system matrix for red light
                system.MG = element_MG @ system.MG  # Update the system matrix for green light
                system.MB = element_MB @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    elif type == 'FreeSpace':
                system.MR = element.matrix() @ system.MR  # Update the system matrix for red light
                system.MG = element.matrix() @ system.MG  # Update the system matrix for green light
                system.MB = element.matrix() @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    elif type == 'Triplet':
                nR1 = refractive_index(0.6563, material[0])  # Refractive index for red light (656.3 nm) for the first lens
                nR2 = refractive_index(0.6563, material[1])  # Refractive index for red light (656.3 nm) for the second lens
                nR3 = refractive_index(0.6563, material[2])  # Refractive index for red light (656.3 nm) for the third lens
                nG1 = refractive_index(0.5876, material[0])  # Refractive index for green light (587.6 nm) for the first lens
                nG2 = refractive_index(0.5876, material[1])  # Refractive index for green light (587.6 nm) for the second lens
                nG3 = refractive_index(0.5876, material[2])  # Refractive index for green light (587.6 nm) for the third lens
                nB1 = refractive_index(0.4861, material[0])  # Refractive index for blue light (486.1 nm) for the first lens
                nB2 = refractive_index(0.4861, material[1])  # Refractive index for blue light (486.1 nm) for the second lens
                nB3 = refractive_index(0.4861, material[2])  # Refractive index for blue light (486.1 nm) for the third lens
                element_MR = triplet_matrix(1, nR1, nR2, nR3, element.R1, element.R2, element.R3, element.R4, element.R5, element.d1, element.d2, element.d3)  # Transfer matrix for red light
                element_MG = triplet_matrix(1, nG1, nG2, nG3, element.R1, element.R2, element.R3, element.R4, element.R5, element.d1, element.d2, element.d3)  # Transfer matrix for green light
                element_MB = triplet_matrix(1, nB1, nB2, nB3, element.R1, element.R2, element.R3, element.R4, element.R5, element.d1, element.d2, element.d3)  # Transfer matrix for blue light
                system.MR = element_MR @ system.MR  # Update the system matrix for red light
                system.MG = element_MG @ system.MG  # Update the system matrix for green light
                system.MB = element_MB @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    else:
                raise ValueError("Element type not recognized")
    return