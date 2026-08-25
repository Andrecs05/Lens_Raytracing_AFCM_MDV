import numpy as np
from scipy.interpolate import griddata
from src.utilities import *
from src.refractive_idxs import *
from src.matrix_formation import *
from src.elements import *
from scipy.ndimage import distance_transform_edt

def _interpolate_sparse_channel(channel_array, min_intensity=0, fill_radius_px=3):
    channel = channel_array.astype(np.float32)
    valid_mask = channel > min_intensity

    if np.count_nonzero(valid_mask) < 3:
        return np.clip(channel, 0, 255).astype(np.uint8)

    points = np.argwhere(valid_mask)
    values = channel[valid_mask]
    grid_y, grid_x = np.indices(channel.shape)

    interpolated = griddata(
        points,
        values,
        (grid_y, grid_x),
        method="linear",
        fill_value=0.0,
    )

    interpolated = np.clip(interpolated, 0, None)

    distance = distance_transform_edt(~valid_mask)
    interpolated[distance > fill_radius_px] = 0.0

    input_values = channel[channel > min_intensity]
    output_values = interpolated[interpolated > 0]

    if input_values.size and output_values.size:
        input_peak = np.percentile(input_values, 99)
        output_peak = np.percentile(output_values, 99)

        if output_peak > 0:
            interpolated *= input_peak / output_peak

    return np.clip(interpolated, 0, 255).astype(np.uint8)

class OpticalSystem:
    def __init__(self, color=False):
        self.color = color  
        if self.color:  # If the system is set to handle color, initialize wavelength-dependent properties here
            # Initialize separate system matrices for red, green, and blue light to account for chromatic effects
            self.MR = np.eye(2)  
            self.MG = np.eye(2) 
            self.MB = np.eye(2)  
            self.MRGB = [self.MR, self.MG, self.MB]  # List to hold the system matrices for each color
            self.sensor = None
        else:
            self.M = np.eye(2)  # Initialize the system matrix for monochromatic light
            self.sensor = None 

    def add_element(self, element, material=['NBK7']):
        if self.color:  # If the system is set to handle color, implement wavelength-dependent behavior here
            if isinstance(element, Sensor):  # If the element is a thick lens, update the system matrices for each color based on the lens material
                self.sensor = element
            else:
                update_elements_color(self, element, material)  # Update the system matrices for each color based on the new element and its material
        else:        
            if isinstance(element, Sensor):  # If the element is a sensor, store it for later use
                self.sensor = element
            else:
                self.M = element.matrix() @ self.M  

    def focal_length(self):
        if self.color:  # If the system is set to handle color, calculate focal lengths for each color separately
            focal_length_R = 1 / (-self.MR[1,0])  
            focal_length_G = 1 / (-self.MG[1,0]) 
            focal_length_B = 1 / (-self.MB[1,0]) 
            return focal_length_R, focal_length_G, focal_length_B 
        else:
            return 1 / (-self.M[1,0])  # Calculate the focal length of the system using the system matrix
        
    def magnification(self):
        if self.color:  # If the system is set to handle color, calculate magnifications for each color separately
            magnification_R = self.MR[0,0] 
            magnification_G = self.MG[0,0] 
            magnification_B = self.MB[0,0]  
            return magnification_R, magnification_G, magnification_B  
        else:
            return self.M[0,0]  # Calculate the magnification of the system using the system matrix
        
    def single_ray_transfer(self, ray, matrix=None, infinity=False):
        if not infinity:
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
        else:  # If the ray is coming from infinity, only the angle matters for the transfer
            ux = ray[0]
            uy = ray[1]
            intensity = ray[2] 
            ray_in_x = np.array([0, ux])
            ray_in_y = np.array([0, uy])
            ray_out_x = matrix @ ray_in_x
            ray_out_y = matrix @ ray_in_y
            output_ray = (ray_out_x[1], ray_out_y[1], intensity)
            return output_ray
        
    def image(self, image_array, pixel_size):
        if self.color:  # If the system is set to handle color, process the image for each color channel separately
            rays_R = image_to_rays_0_angle(image_array, pixel_size, channel=0) 
            rays_G = image_to_rays_0_angle(image_array, pixel_size, channel=1)  
            rays_B = image_to_rays_0_angle(image_array, pixel_size, channel=2)
            
            output_rays_R = [self.single_ray_transfer(ray, self.MR) for ray in rays_R] 
            output_rays_G = [self.single_ray_transfer(ray, self.MG) for ray in rays_G] 
            output_rays_B = [self.single_ray_transfer(ray, self.MB) for ray in rays_B]  
            
            output_image_array_R = rays_to_image(output_rays_R, pixel_size)
            output_image_array_G = rays_to_image(output_rays_G, pixel_size) 
            output_image_array_B = rays_to_image(output_rays_B, pixel_size) 
            
            target_h = max(output_image_array_R.shape[0], output_image_array_G.shape[0], output_image_array_B.shape[0])
            target_w = max(output_image_array_R.shape[1], output_image_array_G.shape[1], output_image_array_B.shape[1])

            output_image_array_R = pad_to_shape(output_image_array_R, target_h, target_w)
            output_image_array_G = pad_to_shape(output_image_array_G, target_h, target_w)
            output_image_array_B = pad_to_shape(output_image_array_B, target_h, target_w)
            
            output_image_array = np.stack((output_image_array_R, output_image_array_G, output_image_array_B), axis=-1)  
            return output_image_array  # Return the resulting image array after processing through the system
        else:
            rays = image_to_rays_0_angle(image_array, pixel_size)  
            output_rays = [self.single_ray_transfer(ray, self.M) for ray in rays] 
            output_image_array = rays_to_image(output_rays, pixel_size)  
            return output_image_array  # Return the resulting image array after processing through the system
        
    def image_object(self, object, pupil_radius, n_rays_per_pixel=7, interpolation=False, min_intensity=30):
        if self.color:  # If the system is set to handle color, process the object for each color channel separately
            rays_R = object_rays(object, pupil_radius, n_rays_per_pixel, channel=0) 
            rays_G = object_rays(object, pupil_radius, n_rays_per_pixel, channel=1)  
            rays_B = object_rays(object, pupil_radius, n_rays_per_pixel, channel=2)

            initial_propagation = FreeSpace(object.distance)
 
            rays_R = [self.single_ray_transfer(ray, initial_propagation.matrix()) for ray in rays_R]
            rays_R = [ray for ray in rays_R if np.sqrt(ray[0]**2 + ray[1]**2) <= pupil_radius]  # Filter rays that are within the pupil radius
            rays_G = [self.single_ray_transfer(ray, initial_propagation.matrix()) for ray in rays_G]
            rays_G = [ray for ray in rays_G if np.sqrt(ray[0]**2 + ray[1]**2) <= pupil_radius]  # Filter rays that are within the pupil radius
            rays_B = [self.single_ray_transfer(ray, initial_propagation.matrix()) for ray in rays_B]
            rays_B = [ray for ray in rays_B if np.sqrt(ray[0]**2 + ray[1]**2) <= pupil_radius]  # Filter rays that are within the pupil radius
            
            output_rays_R = [self.single_ray_transfer(ray, self.MR) for ray in rays_R] 
            output_rays_G = [self.single_ray_transfer(ray, self.MG) for ray in rays_G] 
            output_rays_B = [self.single_ray_transfer(ray, self.MB) for ray in rays_B]  
            
            output_image_array_R = rays_to_image(output_rays_R, M=self.MR, sensor=self.sensor)
            output_image_array_G = rays_to_image(output_rays_G, M=self.MG, sensor=self.sensor)
            output_image_array_B = rays_to_image(output_rays_B, M=self.MB, sensor=self.sensor)

            target_h = max(output_image_array_R.shape[0], output_image_array_G.shape[0], output_image_array_B.shape[0])
            target_w = max(output_image_array_R.shape[1], output_image_array_G.shape[1], output_image_array_B.shape[1])

            output_image_array_R = pad_to_shape(output_image_array_R, target_h, target_w)
            output_image_array_G = pad_to_shape(output_image_array_G, target_h, target_w)
            output_image_array_B = pad_to_shape(output_image_array_B, target_h, target_w)
            
            output_image_array = np.stack((output_image_array_R, output_image_array_G, output_image_array_B), axis=-1)  

            if interpolation:
                interpolated_image_array_R = _interpolate_sparse_channel(output_image_array[:, :, 0], min_intensity=min_intensity)
                interpolated_image_array_G = _interpolate_sparse_channel(output_image_array[:, :, 1], min_intensity=min_intensity)
                interpolated_image_array_B = _interpolate_sparse_channel(output_image_array[:, :, 2], min_intensity=min_intensity)
                output_image_array = np.stack((interpolated_image_array_R, interpolated_image_array_G, interpolated_image_array_B), axis=-1)
            return output_image_array  # Return the resulting image array after processing through the system
        else:
            rays = object_rays(object, pupil_radius, n_rays_per_pixel)
            initial_propagation = FreeSpace(object.distance)
            rays = [self.single_ray_transfer(ray, initial_propagation.matrix()) for ray in rays]
            rays = [ray for ray in rays if np.sqrt(ray[0]**2 + ray[1]**2) <= pupil_radius]  # Filter rays that are within the pupil radius
            output_rays = [self.single_ray_transfer(ray, self.M) for ray in rays] 
            output_image_array = rays_to_image(output_rays, M=self.M, sensor=self.sensor)  

            if interpolation:
                return _interpolate_sparse_channel(output_image_array, min_intensity=min_intensity)

            return output_image_array  # Return the resulting image array after processing through the system

    def image_infinity(self, image_array):
        if self.color:  # If the system is set to handle color, process the image for each color channel separately
            rays_R, cx_R, cy_R = image_infinity_to_rays(image_array, channel=0) 
            rays_G, cx_G, cy_G = image_infinity_to_rays(image_array, channel=1)  
            rays_B, cx_B, cy_B = image_infinity_to_rays(image_array, channel=2)
            
            output_rays_R = [self.single_ray_transfer(ray, self.MR, infinity=True) for ray in rays_R] 
            output_rays_G = [self.single_ray_transfer(ray, self.MG, infinity=True) for ray in rays_G] 
            output_rays_B = [self.single_ray_transfer(ray, self.MB, infinity=True) for ray in rays_B]  
            
            output_image_array_R = rays_to_image_infinity(output_rays_R, cx_R, cy_R)
            output_image_array_G = rays_to_image_infinity(output_rays_G, cx_G, cy_G) 
            output_image_array_B = rays_to_image_infinity(output_rays_B, cx_B, cy_B) 
            
            target_h = max(output_image_array_R.shape[0], output_image_array_G.shape[0], output_image_array_B.shape[0])
            target_w = max(output_image_array_R.shape[1], output_image_array_G.shape[1], output_image_array_B.shape[1])

            output_image_array_R = pad_to_shape(output_image_array_R, target_h, target_w)
            output_image_array_G = pad_to_shape(output_image_array_G, target_h, target_w)
            output_image_array_B = pad_to_shape(output_image_array_B, target_h, target_w)
            
            output_image_array = np.stack((output_image_array_R, output_image_array_G, output_image_array_B), axis=-1)  
            return output_image_array  # Return the resulting image array after processing through the system
        else:
            rays, cx, cy = image_infinity_to_rays(image_array)  
            output_rays = [self.single_ray_transfer(ray, self.M, infinity=True) for ray in rays] 
            output_image_array = rays_to_image_infinity(output_rays, cx, cy)  
            return output_image_array  # Return the resulting image array after processing through the system

    def image_with_interpolation(self, image_array, pixel_size=None, infinity=False):
        if not infinity:
            output_image_array = self.image(image_array, pixel_size) 
        else:
            output_image_array = self.image_infinity(image_array)

        if self.color:  # If the system is set to handle color, perform interpolation for each color channel separately
            interpolated_image_array_R = _interpolate_sparse_channel(output_image_array[:, :, 0], min_intensity=0)
            interpolated_image_array_G = _interpolate_sparse_channel(output_image_array[:, :, 1], min_intensity=0)
            interpolated_image_array_B = _interpolate_sparse_channel(output_image_array[:, :, 2], min_intensity=0)
            interpolated_image_array = np.stack((interpolated_image_array_R, interpolated_image_array_G, interpolated_image_array_B), axis=-1)  # Combine the color channels into a single image array
            return interpolated_image_array  # Return the interpolated image array after processing through the system
        else:
            return _interpolate_sparse_channel(output_image_array, min_intensity=0)
        
    def image_select(self, pixel_size=None, type='infinity', image_array=None, object=None, pupil_radius=None, n_rays_per_pixel=7, interpolation=False, min_intensity=30):
        if type == 'infinity':
            if interpolation:
                return self.image_with_interpolation(image_array, pixel_size, infinity=True)
            else:
                return self.image_infinity(image_array)
        elif type == 'object':
            return self.image_object(object, pupil_radius, n_rays_per_pixel=n_rays_per_pixel, interpolation=interpolation, min_intensity=min_intensity)
        else:
            print("Invalid type specified. Please choose 'infinity' or 'object'.")
            return 

class GalileanTelescope(OpticalSystem):
    def __init__(self, f1=100, f2=-20, lens1=None, lens2=None, magnification=None, color=False, material1=['NBK7'], material2=['NBK7']):
        super().__init__(color=color)  # Call the constructor of the parent class to initialize the color property and system matrices
        if lens1 is not None and lens2 is not None:  # If specific lens objects are provided, use them to build the system
            self.add_element(lens1, material=material1)
            L = lens1.focal_length() + lens2.focal_length() 
            L_propagation = FreeSpace(L)
            self.add_element(L_propagation)
            self.add_element(lens2, material=material2)  
        else:  
            if magnification is None:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.f1 = f1 
                self.f2 = f2  
                self.L = f1 + f2  
                self.add_element(ThinLens(f1), material=material1)  
                self.add_element(FreeSpace(self.L)) 
                self.add_element(ThinLens(f2), material=material2)  
            else:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.f1 = 100  
                self.f2 = -self.f1 / magnification 
                self.L = self.f1 + self.f2  
                self.add_element(ThinLens(self.f1), material=material1)
                self.add_element(FreeSpace(self.L))
                self.add_element(ThinLens(self.f2), material=material2)

class KeplerianTelescope(OpticalSystem):
    def __init__(self, f1=100, f2=20, lens1=None, lens2=None, magnification=None, color=False, material1=['NBK7'], material2=['NBK7']):
        super().__init__(color=color)  # Call the constructor of the parent class to initialize the color property and system matrices
        if lens1 is not None and lens2 is not None:  # If specific lens objects are provided, use them to build the system
            self.add_element(lens1, material=material1)
            L = lens1.focal_length() + lens2.focal_length() 
            L_propagation = FreeSpace(L)
            self.add_element(L_propagation)
            self.add_element(lens2, material=material2)  
        else:  
            if magnification is None:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.f1 = f1 
                self.f2 = f2  
                self.L = self.f1 + self.f2  
                self.add_element(ThinLens(self.f1), material=material1)  
                self.add_element(FreeSpace(self.L)) 
                self.add_element(ThinLens(self.f2), material=material2)  
            else:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.f1 = 100  
                self.f2 = -self.f1 / magnification 
                self.L = self.f1 + self.f2  
                self.add_element(ThinLens(self.f1), material=material1)
                self.add_element(FreeSpace(self.L))
                self.add_element(ThinLens(self.f2), material=material2)

class BrightFieldMicroscope(OpticalSystem):
    def __init__(self, objective_focal_length=4, eyepiece_focal_length=25, objective=None, eyepiece=None, magnification=None, color=False, material_objective=['NBK7'], material_eyepiece=['NBK7']):
        super().__init__(color=color)  # Call the constructor of the parent class to initialize the color property and system matrices
        if objective is not None and eyepiece is not None:  # If specific lens objects are provided, use them to build the system
            self.add_element(objective, material=material_objective)
            L = objective.focal_length() + eyepiece.focal_length() 
            L_propagation = FreeSpace(L)
            self.add_element(L_propagation)
            self.add_element(eyepiece, material=material_eyepiece)  
        else:  
            if magnification is None:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.objective_focal_length = objective_focal_length 
                self.eyepiece_focal_length = eyepiece_focal_length  
                self.L = self.objective_focal_length + self.eyepiece_focal_length  
                self.add_element(ThinLens(self.objective_focal_length), material=material_objective)  
                self.add_element(FreeSpace(self.L)) 
                self.add_element(ThinLens(self.eyepiece_focal_length), material=material_eyepiece)  
            else:  # If a magnification value is provided, calculate the focal lengths based on the magnification
                self.objective_focal_length = 4  
                self.eyepiece_focal_length = -self.objective_focal_length / magnification 
                self.L = self.objective_focal_length + self.eyepiece_focal_length  
                self.add_element(ThinLens(self.objective_focal_length), material=material_objective)
                self.add_element(FreeSpace(self.L))
                self.add_element(ThinLens(self.eyepiece_focal_length), material=material_eyepiece)