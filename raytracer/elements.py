"""Represents the optical system using optical elements such as refracting surfaces, output plane"""

class OpticalElement:
    "Base class for all Optical elements"
    def intercept(self, ray):
        """Base for the intercept"""
        raise NotImplementedError('intercept() needs to be implemented in derived classes')
    def propagate_ray(self, ray):
        """Base for the propogate method"""
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')


class SphericalRefraction:
    def __init__(self, z_0, aperture, curvature, n_1=1., n_2=1.5)