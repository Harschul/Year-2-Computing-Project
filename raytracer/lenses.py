"""Lens classes for simple ray tracing."""

import numpy as np
from raytracer import elements


def n_check(n, wavelength):
    """
    Return the refractive index for a float or material.
    Args:
        n: Refractive index value or material with a ``ref_index`` method.
        wavelength: Wavelength used to evaluate the refractive index.
    Returns:
        The refractive index at the given wavelength.
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
    wavelength=588e-6,
):
    """
    Return the back focal point of a thick lens.
    Args:
        z_0: z-position of the first lens surface.
        thickness: Distance between the two lens surfaces.
        curvature1: Curvature of the first lens surface.
        curvature2: Curvature of the second lens surface.
        n_inside: Refractive index or material inside the lens.
        n_outside: Refractive index or material outside the lens.
        wavelength: Wavelength used to evaluate refractive indices.
    Returns:
        The z-position of the back focal point.
    """
    n1 = n_check(n_inside, wavelength)
    n2 = n_check(n_outside, wavelength)
    n_ratio = n1 / n2
    optical_power = (n_ratio - 1) * (
        curvature1
        - curvature2
        + ((n_ratio - 1) * thickness * curvature1 * curvature2) / n_ratio
    )

    if optical_power == 0:
        focal_length = np.inf
    else:
        focal_length = 1 / optical_power

    back_focal_distance = focal_length * (
        1 - ((n_ratio - 1) * thickness * curvature1) / n_ratio
    )
    return z_0 + thickness + back_focal_distance


class PlanoConvex(elements.OpticalElement):
    """Represent a plano-convex lens."""

    def __init__(
        self,
        *,
        z_0=100,
        curvature=0.0,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    ):
        """
        Creates a plano-convex lens.
        Args:
            z_0: z-position of the first lens surface.
            curvature: Curvature of the curved lens surface.
            n_inside: Refractive index or material inside the lens.
            n_outside: Refractive index or material outside the lens.
            thickness: Distance between the two lens surfaces.
            aperture: Maximum aperture radius of the lens.
        """
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
            z_0=self.z_0(),
            aperture=self.aperture(),
            curvature=curvature1,
            n_1=self.__n_outside,
            n_2=self.__n_inside,
        )

        self.__sr_2 = elements.SphericalRefraction(
            z_0=self.z_0() + self.thickness(),
            aperture=self.aperture(),
            curvature=curvature2,
            n_1=self.__n_inside,
            n_2=self.__n_outside,
        )

    def z_0(self):
        """Return the z-position of the first lens surface."""
        return self.__z_0

    def curvature(self):
        """Return the curvature of the curved lens surface."""
        return self.__curvature

    def n_inside(self, wavelength=588e-6):
        """
        Return the refractive index inside the lens.
        Args:
            wavelength: Wavelength used to evaluate the refractive index.
        Returns:
            The refractive index inside the lens.
        """
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength=588e-6):
        """
        Return the refractive index outside the lens.
        Args:
            wavelength: Wavelength used to evaluate the refractive index.

        Returns:
            The refractive index outside the lens.
        """
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Return the lens thickness."""
        return self.__thickness

    def aperture(self):
        """Return the lens aperture."""
        return self.__aperture

    @property
    def sr_1(self):
        """Return the first refracting surface."""
        return self.__sr_1

    @property
    def sr_2(self):
        """Return the second refracting surface."""
        return self.__sr_2

    def focal_point(self, wavelength=588e-6):
        """
        Return the back focal point of the lens.
        Args:
            wavelength: Wavelength used to evaluate refractive indices.
        Returns:
            The z-position of the back focal point.
        """
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
        """
        Return the ray intercept with the first lens surface.
        Args:
            ray: Ray whose intercept should be found.
        Returns:
            The intercept with the first refracting surface.
        """
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """
        Propagate ray thorough the lens.
        The ray is first propagated through the first refracting surface. If
        the ray misses that surface, propagation stops.

        Args:
            ray: Ray to propagate through the lens.

        Returns:
            The propagated ray, or ``None`` if propagation fails.
        """
        og_length = len(ray.vertices())
        self.sr_1.propagate_ray(ray)
        new_length = len(ray.vertices())

        if og_length == new_length:
            return None

        return self.sr_2.propagate_ray(ray)


class BiConvex(elements.OpticalElement):
    """Represents a bi convex lens."""

    def __init__(
        self,
        *,
        z_0=100,
        curvature1=0.02,
        curvature2=-0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    ):
        """
        Create a bi-convex lens.
        Args:
            z_0: z position of the first lens surface.
            curvature1: Curvature of the first lens surface.
            curvature2: Curvature of the second lens surface.
            n_inside: Refractive index or material inside the lens.
            n_outside: Refractive index or material outside the lens.
            thickness: Distance between the two lens surfaces.
            aperture: Maximum aperture radius of the lens.
        """
        self.__z_0 = z_0
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture

        self.__sr_1 = elements.SphericalRefraction(
            z_0=self.z_0(),
            aperture=self.aperture(),
            curvature=self.curvature1(),
            n_1=self.__n_outside,
            n_2=self.__n_inside,
        )

        self.__sr_2 = elements.SphericalRefraction(
            z_0=self.z_0() + self.thickness(),
            aperture=self.aperture(),
            curvature=self.curvature2(),
            n_1=self.__n_inside,
            n_2=self.__n_outside,
        )

    def z_0(self):
        """Return the z-position of the first lens surface."""
        return self.__z_0

    def curvature1(self):
        """Return the curvature of the first lens surface."""
        return self.__curvature1

    def curvature2(self):
        """Return the curvature of the second lens surface."""
        return self.__curvature2

    def n_inside(self, wavelength=588e-6):
        """
        Return the refractive index inside the lens.
        Args:
            wavelength: Wavelength used to evaluate the refractive index.
        Returns:
            The refractive index inside the lens.
        """
        return n_check(self.__n_inside, wavelength)

    def n_outside(self, wavelength=588e-6):
        """
        Return the refractive index outside the lens.
        Args:
            wavelength: Wavelength used to evaluate the refractive index.
        Returns:
            The refractive index outside the lens.
        """
        return n_check(self.__n_outside, wavelength)

    def thickness(self):
        """Return the lens thickness."""
        return self.__thickness

    def aperture(self):
        """Return the lens aperture."""
        return self.__aperture

    @property
    def sr_1(self):
        """Return the first refracting surface."""
        return self.__sr_1

    @property
    def sr_2(self):
        """Return the second refracting surface."""
        return self.__sr_2

    def focal_point(self, wavelength=588e-6):
        """
        Return the back focal point of the lens.
        Args:
            wavelength: Wavelength used to evaluate refractive indices.
        Returns:
            The z-position of the back focal point.
        """
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
        """
        Return the ray intercept with the first lens surface.
        Args:
            ray: Ray whose intercept should be found.
        Returns:
            The intercept with the first refracting surface.
        """
        return self.sr_1.intercept(ray)

    def propagate_ray(self, ray):
        """
        Propagate a rau through both lens surfaces.
        The ray is first propagated through the first refracting surface. If
        the ray misses that surface, propagation stops.
        Args:
            ray: Ray to propagate through the lens.

        Returns:
            The propagated ray, or ``None`` if propagation fails.
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
    """Represent a convex-plano lens."""
