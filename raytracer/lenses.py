"""Module containing all lens classes"""
from raytracer import elements

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
            n_1 = self.n_outside(),
            n_2 = self.n_inside(),
        )

        self.sr_2 = elements.SphericalRefraction(
            z_0 = self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = curvature2,
            n_1 = self.n_inside(),
            n_2 = self.n_outside(),
        )

    def z_0(self):
        """Return the z_0 position"""
        return self.__z_0

    def curvature(self):
        """Return the curvature1"""
        return self.__curvature

    def n_inside(self):
        """Return the refractive index inside the lens"""
        return self.__n_inside

    def n_outside(self):
        """Return the refractive index outside the lens"""
        return self.__n_outside

    def thickness(self):
        """Return the lens thickness"""
        return self.__thickness

    def aperture(self):
        """Return the aperture"""
        return self.__aperture

    def intercept(self, ray):
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """Intercept of one ray with the lens"""
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())
        if og_length == new_length:
            return
        self.sr_2.propagate_ray(ray)
