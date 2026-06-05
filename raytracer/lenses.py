"""Lens classes"""
import numpy as np
from raytracer import elements

def n_check(n, wavelength):
    """
    Returns refractive index from float or material
    Details: material must have ref_index
    """
    if hasattr(n, "ref_index") and callable(n.ref_index):
        return n.ref_index(wavelength)
    return n

def _lensmaker_focal_point(
    z_0,
    thickness,
    curvature1,
    curvature2,
    n_inside,
    n_outside,
    wavelength = 588e-6
):
    """
    Returns back focal point
    Details: uses thick lens Lensmaker equation
    """
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
    """Creates plano convex lens"""
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
        """Initializes plano convex lens"""
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

        self.__sr_1 = elements.SphericalRefraction(
            z_0 = self.z_0(),
            aperture = self.aperture(),
            curvature = curvature1,
            n_1 = self.__n_outside,
            n_2 = self.__n_inside,
        )

        self.__sr_2 = elements.SphericalRefraction(
            z_0 = self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = curvature2,
            n_1 = self.__n_inside,
            n_2 = self.__n_outside,
        )

    def z_0(self):
        """Returns z0 position"""
        return self.__z_0

    def curvature(self):
        """Returns curvature"""
        return self.__curvature

    def n_inside(self, wavelength = 588e-6):
        """
        Returns refractive index inside lens
        Details: default wavelength is 588e-6 mm
        """
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength = 588e-6):
        """
        Returns refractive index outside lens
        Details: default wavelength is 588e-6 mm
        """
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Returns lens thickness"""
        return self.__thickness

    def aperture(self):
        """Returns aperture"""
        return self.__aperture


    @property
    def sr_1(self):
        """Returns first surface"""
        return self.__sr_1

    @property
    def sr_2(self):
        """Returns second surface"""
        return self.__sr_2

    def focal_point(self, wavelength = 588e-6):
        """Returns back focal point"""
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
        """Returns first surface intercept"""
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """Propagates ray through lens"""
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())
        if og_length == new_length:
            return None
        return self.sr_2.propagate_ray(ray)

class BiConvex(elements.OpticalElement):
    """Creates bi convex lens"""
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
        """Initializes bi convex lens"""
        self.__z_0 = z_0
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture

        self.__sr_1 = elements.SphericalRefraction(
            z_0 = self.z_0(),
            aperture = self.aperture(),
            curvature = self.curvature1(),
            n_1 = self.__n_outside,
            n_2 = self.__n_inside,
        )

        self.__sr_2 = elements.SphericalRefraction(
            z_0 = self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = self.curvature2(),
            n_1 = self.__n_inside,
            n_2 = self.__n_outside,
        )

    def z_0(self):
        """Returns z0 position"""
        return self.__z_0

    def curvature1(self):
        """Returns first curvature"""
        return self.__curvature1

    def curvature2(self):
        """Returns second curvature"""
        return self.__curvature2

    def n_inside(self, wavelength = 588e-6):
        """Returns refractive index inside lens"""
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength = 588e-6):
        """Returns refractive index outside lens"""
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Returns lens thickness"""
        return self.__thickness

    def aperture(self):
        """Returns aperture"""
        return self.__aperture


    @property
    def sr_1(self):
        """Returns first surface"""
        return self.__sr_1

    @property
    def sr_2(self):
        """Returns second surface"""
        return self.__sr_2

    def focal_point(self, wavelength = 588e-6):
        """Returns back focal point"""
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
        """Returns first surface intercept"""
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """
        Propagates ray through both lens surfaces
        Details: stops if ray misses first surface
        """
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())
        if og_length == new_length:
            return None
        return self.sr_2.propagate_ray(ray)

# A Convex Plano lens can just be created using the Plano Convex Class.
# This has however been inserted to pass the Advanced Design Test.
class ConvexPlano(PlanoConvex):
    """Creates convex plano lens"""
