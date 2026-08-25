import numpy as np
from numpy.ma import sqrt
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
    cmin = min(cx, cy)  # Determine the maximum center coordinate for normalization
    for i in range(height):
        for j in range(width):
            intensity = image_array[i, j, channel]  # Assuming the first channel represents intensity
            if intensity > 0:  # Only consider pixels with non-zero intensity
                anglex = (j - cx) / cmin
                angley = (i - cy) / cmin
                if abs(anglex) <= 1 and abs(angley) <= 1 and sqrt(anglex**2 + angley**2) <= 1:  # Ensure the angles are within the valid range
                    rays.append((anglex, angley, intensity))  # Append the ray with its intensity for further processing
    return rays, cx, cy

def object_rays(object, pupil_radius, n_rays_per_pixel, channel=0):
    d_obj = object.distance 
    h_obj = object.height 
    array = object.image_array  
    try:
        height, width, channels = array.shape  
    except ValueError:
        height, width = array.shape

    pixel_size = h_obj / array.shape[0]  
    theta_half_width = np.arctan(pupil_radius / d_obj) * 0.5  # half-angle of the cone
    weight = 1 / n_rays_per_pixel  
    rays = []
    for i in range(height):
        for j in range(width):
            try:
                intensity = array[i, j, channel] * weight
            except IndexError:
                intensity = array[i, j] * weight
            if intensity > 0:  
                x = (j - width / 2) * pixel_size  
                y = (i - height / 2) * pixel_size  
                theta_chief = np.arctan(np.sqrt(x**2 + y**2) / d_obj)  # chief ray angle
                if n_rays_per_pixel == 1:
                    thetas = np.array([theta_chief])  # only the chief ray
                else:
                    #thetas = np.linspace(-theta_half_width, theta_half_width, n_rays_per_pixel)  # include 0 angle and random angles within the cone
                    thetas = np.concatenate(([0], [theta_chief], theta_chief + np.random.uniform(-theta_half_width, theta_half_width, n_rays_per_pixel - 2)))  # include 0 angle and random angles within the cone
                thetas = thetas
                for theta in thetas:
                    rays.append((x, y, theta, intensity))
    return rays





def rays_to_image(rays, pixel_size=None, M=None, Obj_height=None, sensor=None):
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
    # Determine the size of the image based on the magnification
    Mag = abs(M[0, 0]) if M is not None else 1  # Use the magnification from the transfer matrix if provided
    if Obj_height is not None or sensor is not None:
        if sensor is not None:
            pixel_size = sensor.pixel_size
            width, height = sensor.resolution
        else:
            max_r = int(np.ceil(Obj_height * Mag / pixel_size)) + 1  # Calculate the height of the image array based on object height and magnification
            width = max_r
            height = max_r
        image_array = np.zeros((height, width), dtype=np.float32)  # Initialize the image array with zeros
        for ray in rays:
            x_centered, y_centered, angle, intensity = ray
            x = x_centered + (width * pixel_size) / 2  # Adjust x-coordinate back to image space
            y = y_centered + (height * pixel_size) / 2  # Adjust y-coordinate back to image space
            j = int(x / pixel_size)  # Calculate the pixel index for x-coordinate
            i = int(y / pixel_size)  # Calculate the pixel index for y-coordinate
            if 0 <= i < height and 0 <= j < width:  # Ensure the indices are within the bounds of the image array
                image_array[i, j] += intensity  # Add the intensity of the ray to the corresponding pixel
    # else:
    #     max_x = max(ray[0] for ray in rays)
    #     max_y = max(ray[1] for ray in rays)
    #     width = int(np.ceil(max_x * 2/ pixel_size)) + 1  # Calculate the width of the image array
    #     height = int(np.ceil(max_y * 2/ pixel_size)) + 1  # Calculate the height of the image array
    #     image_array = np.zeros((height, width), dtype=np.float32)  # Initialize the image array with zeros
    #     for ray in rays:
    #         x_centered, y_centered, angle, intensity = ray
    #         x = x_centered + (width * pixel_size) / 2  # Adjust x-coordinate back to image space
    #         y = y_centered + (height * pixel_size) / 2  # Adjust y-coordinate back to image space
    #         j = int(x / pixel_size)  # Calculate the pixel index for x-coordinate
    #         i = int(y / pixel_size)  # Calculate the pixel index for y-coordinate
    #         if 0 <= i < height and 0 <= j < width:  # Ensure the indices are within the bounds of the image array
    #             image_array[i, j] += intensity  # Add the intensity of the ray to the corresponding pixel

    else:
        if len(rays) == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        max_x = max(abs(ray[0]) for ray in rays)
        max_y = max(abs(ray[1]) for ray in rays)
        width = int(np.ceil(max_x * 2 / pixel_size)) + 2
        height = int(np.ceil(max_y * 2 / pixel_size)) + 2

    # float accumulation buffer
    image_array = np.zeros((height, width), dtype=np.float32)

    # Bilinear splat: each ray contributes to 4 neighboring pixels
    for ray in rays:
        x_centered, y_centered, angle, intensity = ray

        u = (x_centered / pixel_size) + (width / 2.0)
        v = (y_centered / pixel_size) + (height / 2.0)

        j0 = int(np.floor(u))
        i0 = int(np.floor(v))
        du = u - j0
        dv = v - i0

        # neighbors and bilinear weights
        nbrs = [
            (i0,     j0,     (1 - du) * (1 - dv)),
            (i0,     j0 + 1, du * (1 - dv)),
            (i0 + 1, j0,     (1 - du) * dv),
            (i0 + 1, j0 + 1, du * dv),
        ]

        for ii, jj, w in nbrs:
            if 0 <= ii < height and 0 <= jj < width and w > 0:
                image_array[ii, jj] += intensity * w

    # Robust display normalization (avoids one outlier making whole image too dark)
    nz = image_array[image_array > 0]
    if nz.size == 0:
        return np.zeros_like(image_array, dtype=np.uint8)

    scale = np.percentile(nz, 99.5)
    if scale <= 0:
        return np.zeros_like(image_array, dtype=np.uint8)

    image_array = np.clip(image_array / scale, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    return image_array
    
    # Normalize the image array to the range [0, 255] for visualization purposes
    # image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
    # return image_array

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
    cmin = min(cx, cy)  # Determine the minimum center coordinate for normalization
    max_ux = max(abs(ray[0]) for ray in rays)
    max_uy = max(abs(ray[1]) for ray in rays)
    j_max = (int(np.ceil(max_ux * cmin + cmin)) + 1) * 2  # Calculate the maximum pixel index for x-direction
    i_max = (int(np.ceil(max_uy * cmin + cmin)) + 1) * 2  # Calculate the maximum pixel index for y-direction
    image_array = np.zeros((i_max, j_max), dtype=np.float32)  # Initialize the image array with zeros
    for ray in rays:
        ux_out, uy_out, intensity = ray
        x = int((ux_out + 1) * cmin) + j_max // 2  # Convert angle back to pixel index for x-coordinate
        y = int((uy_out + 1) * cmin) + i_max // 2  # Convert angle back to pixel index for y-coordinate
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
    elif type == 'ThinLens':
                system.MR = element.matrix() @ system.MR  # Update the system matrix for red light
                system.MG = element.matrix() @ system.MG  # Update the system matrix for green light
                system.MB = element.matrix() @ system.MB  # Update the system matrix for blue light
                system.MRGB = [system.MR, system.MG, system.MB]  # Update the list of system matrices for each color
    else:
                raise ValueError("Element type not recognized")
    return