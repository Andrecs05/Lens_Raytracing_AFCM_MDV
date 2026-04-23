import numpy as np
from scipy.interpolate import griddata
from src.utilities2 import *
from src.refractive_idxs import *
from src.matrix_formation import *

class OpticalSystem:
    def __init__(self, color=False):
        self.color = color  
        if self.color:  # If the system is set to handle color, we can initialize wavelength-dependent properties here
            self.MR = np.eye(2)  # Initialize the system matrix for red light
            self.MG = np.eye(2)  # Initialize the system matrix for green light
            self.MB = np.eye(2)  # Initialize the system matrix for blue light
            self.MRGB = [self.MR, self.MG, self.MB]  # List to hold the system matrices for each color
        else:
            self.M = np.eye(2)  # Initialize the system matrix for monochromatic light
    def add_element(self, element, material='NBK7'):
        if self.color:  # If the system is set to handle color, we can implement wavelength-dependent behavior here
            update_elements_color(self, element, material)  # Update the system matrices for each color based on the new element and its material
        else:        
            self.M = element.matrix() @ self.M  # Update the system matrix by multiplying with the new element's matrix
    def focal_length(self):
        if self.color:  # If the system is set to handle color, we can calculate focal lengths for each color separately
            focal_length_R = 1 / (-self.MR[1,0])  # Calculate the focal length for red light using the red system matrix
            focal_length_G = 1 / (-self.MG[1,0])  # Calculate the focal length for green light using the green system matrix
            focal_length_B = 1 / (-self.MB[1,0])  # Calculate the focal length for blue light using the blue system matrix
            return focal_length_R, focal_length_G, focal_length_B  # Return the focal lengths for each color as a tuple
        else:
            return 1 / (-self.M[1,0])  # Calculate the focal length of the system using the system matrix
    def magnification(self):
        if self.color:  # If the system is set to handle color, we can calculate magnifications for each color separately
            magnification_R = self.MR[0,0]  # Calculate the magnification for red light using the red system matrix
            magnification_G = self.MG[0,0]  # Calculate the magnification for green light using the green system matrix
            magnification_B = self.MB[0,0]  # Calculate the magnification for blue light using the blue system matrix
            return magnification_R, magnification_G, magnification_B  # Return the magnifications for each color as a tuple
        else:
            return self.M[0,0]  # Calculate the magnification of the system using the system matrix
    def single_ray_transfer(self, ray, matrix=None):
        if len(ray) == 4:  # If the ray includes intensity, ignore it for the transfer
            r = np.sqrt(ray[0]**2 + ray[1]**2) 
            phi = np.arctan2(ray[1], ray[0]) 
            theta = ray[2] 
            intensity = ray[3] 
            ray_vector = np.array([r, theta]) 
            output_vector = matrix @ ray_vector 
            x_output = output_vector[0] * np.cos(phi)  
            y_output = output_vector[0] * np.sin(phi)  
            output_ray = (x_output, y_output, output_vector[1], intensity) 
            return output_ray  
        else:  # If the ray does not include intensity, just transfer the ray vector
            r = np.sqrt(ray[0]**2 + ray[1]**2)
            phi = np.arctan2(ray[1], ray[0]) 
            theta = ray[2]  
            ray_vector = np.array([r, theta])  
            output_vector = matrix @ ray_vector  
            x_output = output_vector[0] * np.cos(phi)
            y_output = output_vector[0] * np.sin(phi)
            output_ray = (x_output, y_output, output_vector[1])
            return output_ray
    def image(self, image_array, pixel_size):
        if self.color:  # If the system is set to handle color, we can process the image for each color channel separately
            rays_R = image_to_rays(image_array, pixel_size, channel=0)  # Convert the input image array into rays for the red channel
            rays_G = image_to_rays(image_array, pixel_size, channel=1)  # Convert the input image array into rays for the green channel
            rays_B = image_to_rays(image_array, pixel_size, channel=2)  # Convert the input image array into rays for the blue channel
            
            output_rays_R = [self.single_ray_transfer(ray, self.MR) for ray in rays_R]  # Transfer each ray through the system for the red channel
            output_rays_G = [self.single_ray_transfer(ray, self.MG) for ray in rays_G]  # Transfer each ray through the system for the green channel
            output_rays_B = [self.single_ray_transfer(ray, self.MB) for ray in rays_B]  # Transfer each ray through the system for the blue channel
            
            output_image_array_R = rays_to_image(output_rays_R, pixel_size)  # Convert the output rays back into an image array for the red channel
            output_image_array_G = rays_to_image(output_rays_G, pixel_size)  # Convert the output rays back into an image array for the green channel
            output_image_array_B = rays_to_image(output_rays_B, pixel_size)  # Convert the output rays back into an image array for the blue channel
            
            target_h = max(output_image_array_R.shape[0], output_image_array_G.shape[0], output_image_array_B.shape[0])
            target_w = max(output_image_array_R.shape[1], output_image_array_G.shape[1], output_image_array_B.shape[1])

            output_image_array_R = pad_to_shape(output_image_array_R, target_h, target_w)
            output_image_array_G = pad_to_shape(output_image_array_G, target_h, target_w)
            output_image_array_B = pad_to_shape(output_image_array_B, target_h, target_w)
            
            output_image_array = np.stack((output_image_array_R, output_image_array_G, output_image_array_B), axis=-1)  # Combine the color channels into a single image array
            return output_image_array  # Return the resulting image array after processing through the system
        else:
            rays = image_to_rays(image_array, pixel_size)  # Convert the input image array into rays
            output_rays = [self.single_ray_transfer(ray, self.M) for ray in rays]  # Transfer each ray through the system
            output_image_array = rays_to_image(output_rays, pixel_size)  # Convert the output rays back into an image array
            return output_image_array  # Return the resulting image array after processing through the system
    def image_with_interpolation(self, image_array, pixel_size):
        if self.color:  # If the system is set to handle color, we can process the image for each color channel separately
            rays_R = image_to_rays(image_array, pixel_size, channel=0)  # Convert the input image array into rays for the red channel
            rays_G = image_to_rays(image_array, pixel_size, channel=1)  # Convert the input image array into rays for the green channel
            rays_B = image_to_rays(image_array, pixel_size, channel=2)  # Convert the input image array into rays for the blue channel

            output_rays_R = [self.single_ray_transfer(ray, self.MR) for ray in rays_R]  # Transfer each ray through the system for the red channel
            output_rays_G = [self.single_ray_transfer(ray, self.MG) for ray in rays_G]  # Transfer each ray through the system for the green channel
            output_rays_B = [self.single_ray_transfer(ray, self.MB) for ray in rays_B]  # Transfer each ray through the system for the blue channel

            output_image_array_R = rays_to_image(output_rays_R, pixel_size)  # Convert the output rays back into an image array for the red channel
            output_image_array_G = rays_to_image(output_rays_G, pixel_size)  # Convert the output rays back into an image array for the green channel
            output_image_array_B = rays_to_image(output_rays_B, pixel_size)  # Convert the output rays back into an image array for the blue channel

            target_h = max(output_image_array_R.shape[0], output_image_array_G.shape[0], output_image_array_B.shape[0])
            target_w = max(output_image_array_R.shape[1], output_image_array_G.shape[1], output_image_array_B.shape[1])

            output_image_array_R = pad_to_shape(output_image_array_R, target_h, target_w)
            output_image_array_G = pad_to_shape(output_image_array_G, target_h, target_w)
            output_image_array_B = pad_to_shape(output_image_array_B, target_h, target_w)
            output_image_array = np.stack((output_image_array_R, output_image_array_G, output_image_array_B), axis=-1)  # Combine the color channels into a single image array
            
            interpolated_image_array_R = griddata(np.argwhere(output_image_array_R>0), output_image_array_R[output_image_array_R>0], (np.indices(output_image_array_R.shape)[0], np.indices(output_image_array_R.shape)[1]), method='cubic', fill_value=0)  # Perform cubic interpolation for the red channel
            interpolated_image_array_G = griddata(np.argwhere(output_image_array_G>0), output_image_array_G[output_image_array_G>0], (np.indices(output_image_array_G.shape)[0], np.indices(output_image_array_G.shape)[1]), method='cubic', fill_value=0)  # Perform cubic interpolation for the green channel
            interpolated_image_array_B = griddata(np.argwhere(output_image_array_B>0), output_image_array_B[output_image_array_B>0], (np.indices(output_image_array_B.shape)[0], np.indices(output_image_array_B.shape)[1]), method='cubic', fill_value=0)  # Perform cubic interpolation for the blue channel
            
            interpolated_image_array_R = (interpolated_image_array_R / np.max(interpolated_image_array_R)) * np.max(image_array[:,:,0])  # Normalize the interpolated red channel
            interpolated_image_array_G = (interpolated_image_array_G / np.max(interpolated_image_array_G)) * np.max(image_array[:,:,1])  # Normalize the interpolated green channel
            interpolated_image_array_B = (interpolated_image_array_B / np.max(interpolated_image_array_B)) * np.max(image_array[:,:,2])  # Normalize the interpolated blue channel
            
            interpolated_image_array_R = np.clip(interpolated_image_array_R, 0, 255).astype(np.uint8)  # Clip values to the range [0, 255] and convert to uint8 for the red channel
            interpolated_image_array_G = np.clip(interpolated_image_array_G, 0, 255).astype(np.uint8)  # Clip values to the range [0, 255] and convert to uint8 for the green channel
            interpolated_image_array_B = np.clip(interpolated_image_array_B, 0, 255).astype(np.uint8)  # Clip values to the range [0, 255] and convert to uint8 for the blue channel
            
            interpolated_image_array = np.stack((interpolated_image_array_R, interpolated_image_array_G, interpolated_image_array_B), axis=-1)  # Combine the color channels into a single image array
            return interpolated_image_array  # Return the interpolated image array after processing through the system
        else:
            rays = image_to_rays(image_array, pixel_size)  # Convert the input image array into rays
            output_rays = [self.single_ray_transfer(ray, self.M) for ray in rays]  # Transfer each ray through the system
            output_image_array = rays_to_image(output_rays, pixel_size)  # Convert the output rays back into an image array
            # Interpolate the output image array to fill in any gaps
            interpolated_image_array = griddata(np.argwhere(output_image_array>0), output_image_array[output_image_array>0], (np.indices(output_image_array.shape)[0], np.indices(output_image_array.shape)[1]), method='cubic', fill_value=0)  # Perform cubic interpolation
            # Normalize the interpolated image array and convert it back to unsigned 8-bit integer format for visualization
            interpolated_image_array = (interpolated_image_array / np.max(interpolated_image_array)) * np.max(image_array)
            interpolated_image_array = np.clip(interpolated_image_array, 0, 255).astype(np.uint8)  # Clip values to the range [0, 255] and convert to uint8
            return interpolated_image_array  # Return the interpolated image array after processing through the system