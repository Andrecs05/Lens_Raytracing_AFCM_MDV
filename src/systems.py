import numpy as np

class OpticalSystem:
    def __init__(self):
        self.M = np.eye(2)  # Initialize the system matrix as the identity matrix
    def add_element(self, element):
        self.M = element.matrix() @ self.M  # Update the system matrix by multiplying with the new element's matrix
    def focal_length(self):
        return 1 / (-self.M[1,0])  # Calculate the focal length of the system using the system matrix
    def magnification(self):
        return self.M[0,0]  # Calculate the magnification of the system using the system matrix
