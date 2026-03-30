import numpy as np
from .matrix_formation import *

class ThickLens:
    def __init__(self, R1, R2, d, n):
        '''
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        d : float - Thickness of the lens
        n : float - Refractive index of the lens material
        '''
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.n = n
    def matrix(self):           # Calculate the transfer matrix for a thick lens using the provided parameters
        return thick_lens_transfer_matrix(self.R1, self.R2, self.d, self.n)
    def focal_length(self):     # Calculate the focal length of the thick lens using the transfer matrix
        return 1 / (-self.matrix()[1,0])

class ThinLens:
    def __init__(self, f):
        '''
        f : float - Focal length of the lens
        '''
        self.f = f
    def matrix(self):       # Calculate the transfer matrix for a thin lens using the provided focal length
        return thin_lens_transfer_matrix(self.f)

class Mirror:
    def __init__(self, R):
        '''
        R : float - Radius of curvature (positive if center to the right)
        '''
        self.R = R
    def matrix(self):       # Calculate the reflection matrix for a spherical mirror using the provided radius of curvature
        return reflection_matrix(self.R)
    
class Doublet:
    def __init__(self, R1, R2, R3, d1, d2, n1, n2):
        '''
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        R3 : float - Radius of curvature of the third surface (positive if center to the right)
        d1 : float - Thickness of the first lens
        d2 : float - Thickness of the second lens
        n1 : float - Refractive index of the first lens material
        n2 : float - Refractive index of the second lens material
        '''
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.d1 = d1
        self.d2 = d2
        self.n1 = n1
        self.n2 = n2
    def matrix(self):        # Calculate the transfer matrix for the doublet using the provided parameters
        return doublet_matrix(self.R1, self.R2, self.R3, self.d1, self.d2, self.n1, self.n2)
    def focal_length(self):  # Calculate the focal length of the doublet using the transfer matrix
        return 1 / (-self.matrix()[1,0])
    
class Triplet:
    def __init__(self, R1, R2, R3, R4, d1, d2, d3, n1, n2, n3):
        '''
        R1 : float - Radius of curvature of the first surface (positive if center to the right)
        R2 : float - Radius of curvature of the second surface (positive if center to the right)
        R3 : float - Radius of curvature of the third surface (positive if center to the right)
        R4 : float - Radius of curvature of the fourth surface (positive if center to the right)
        d1 : float - Thickness of the first lens
        d2 : float - Thickness of the second lens
        d3 : float - Thickness of the third lens
        n1 : float - Refractive index of the first lens material
        n2 : float - Refractive index of the second lens material
        n3 : float - Refractive index of the third lens material
        '''
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
    def matrix(self):       # Calculate the transfer matrix for the triplet using the provided parameters
        return triplet_matrix(self.R1, self.R2, self.R3, self.R4, self.d1, self.d2, self.d3, self.n1, self.n2, self.n3)
    def focal_length(self): # Calculate the focal length of the triplet using the transfer matrix
        return 1 / (-self.matrix()[1,0])
    
class FreeSpace:
    def __init__(self, d, n=1):
        '''
        d : float - Distance of free space
        '''
        self.d = d
        self.n = n
    def matrix(self):       # Calculate the transfer matrix for free space using the provided distance and refractive index
        return translation_matrix(self.d, self.n)