"""Module containing all lens classes"""
import numpy as np
from raytracer import elements

def n_check(n, wavelength):
    """Return refractive index from either a float or a material"""
    if hasattr(n, "ref_index") and callable(n.ref_index):
        return n.ref_index(wavelength)

    return n

def _lensmaker_focal_point(z_0, thickness, curvature1, curvature2, n_inside, n_outside, wavelength = 588e-6):
    """Return the back focal point using the thick lens Lensmaker equation"""
    n1 = n_check(n_inside, wavelength)
    n2 = n_check(n_outside, wavelength)
    n_ratio = n1 / n2
    optical_power = (n_ratio - 1) * (curvature1 - curvature2 + ((n_ratio - 1) * thickness * curvature1 * curvature2) / n_ratio)
    if optical_power == 0:
        focal_length = np.inf
    else:
        focal_length = 1 / optical_power
    back_focal_distance = focal_length * (1 - ((n_ratio - 1) * thickness * curvature1) / n_ratio)
    return z_0 + thickness + back_focal_distance


class PlanoConvex(elements.OpticalElement):
    """Creates PlanoConvex lens"""
    def __init__(
        self,
        *,
        z_0 = 100,
        curvature = 0.0,
        n_inside = 1.5168,
        n_outside = 1.0,
        thickness = 5.0,
        aperture = 50.0,
    ):
        """Initializes the Plano Convex Lens"""
        self.__z_0 = z_0
        self.__curvature = curvature
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture

        if self.curvature() < 0:
            curvature1 = 0
            curvature2 = self.curvature()
        else:
            curvature1 = self.curvature()
            curvature2 = 0

        self.sr_1 = elements.SphericalRefraction(
            z_0 = self.z_0(),
            aperture = self.aperture(),
            curvature = curvature1,
            n_1 = self.__n_outside,
            n_2 = self.__n_inside,
        )

        self.sr_2 = elements.SphericalRefraction(
            z_0 = self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = curvature2,
            n_1 = self.__n_inside,
            n_2 = self.__n_outside,
        )

    def z_0(self):
        """Return the z_0 position"""
        return self.__z_0

    def curvature(self):
        """Return the curvature1"""
        return self.__curvature

    def n_inside(self, wavelength = 588e-6):
        """Return the refractive index inside the lens"""
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength = 588e-6):
        """Return the refractive index outside the lens"""
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Return the lens thickness"""
        return self.__thickness

    def aperture(self):
        """Return the aperture"""
        return self.__aperture

    def focal_point(self, wavelength = 588e-6):
        """Return the back focal point of the lens"""
        return _lensmaker_focal_point(
            self.z_0(),
            self.thickness(),
            self.sr_1.curvature(),
            self.sr_2.curvature(),
            self.__n_inside,
            self.__n_outside,
            wavelength,
        )

    def intercept(self, ray):
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """Intercept of one ray with the lens"""
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())
        if og_length == new_length:
            return None
        return self.sr_2.propagate_ray(ray)

class BiConvex(elements.OpticalElement):
    """Creates BiConvex lens"""
    def __init__(
        self,
        *,
        z_0 = 100,
        curvature1 = 0.02,
        curvature2 = -0.02,
        n_inside = 1.5168,
        n_outside = 1.0,
        thickness = 5.0,
        aperture = 50.0,
    ):
        """Initializes the Bi Convex Lens"""
        self.__z_0 = z_0
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture

        self.sr_1 = elements.SphericalRefraction(
            z_0 = self.z_0(),
            aperture = self.aperture(),
            curvature = self.curvature1(),
            n_1 = self.__n_outside,
            n_2 = self.__n_inside,
        )

        self.sr_2 = elements.SphericalRefraction(
            z_0 = self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = self.curvature2(),
            n_1 = self.__n_inside,
            n_2 = self.__n_outside,
        )

    def z_0(self):
        """Return the z_0 position"""
        return self.__z_0

    def curvature1(self):
        """Return the curvature1"""
        return self.__curvature1

    def curvature2(self):
        """Return the curvature2"""
        return self.__curvature2

    def n_inside(self, wavelength = 588e-6):
        """Return the refractive index inside the lens"""
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength = 588e-6):
        """Return the refractive index outside the lens"""
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Return the lens thickness"""
        return self.__thickness

    def aperture(self):
        """Return the aperture"""
        return self.__aperture

    def focal_point(self, wavelength = 588e-6):
        """Return the back focal point of the lens"""
        return _lensmaker_focal_point(
            self.z_0(),
            self.thickness(),
            self.curvature1(),
            self.curvature2(),
            self.__n_inside,
            self.__n_outside,
            wavelength,
        )

    def intercept(self, ray):
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """Intercept of one ray with the lens"""
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())
        if og_length == new_length:
            return None
        return self.sr_2.propagate_ray(ray)
