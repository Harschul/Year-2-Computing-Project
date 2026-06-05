"""Optical element classes for tracing rays through simple optical surfaces."""

import numpy as np
from raytracer import physics
from raytracer.rays import Ray


class OpticalElement:
    """Represent a base optical element."""
    def intercept(self, ray):
        """
        Return the first intercept between a ray and the optical element.
        Args:
            ray: Ray to intercept with the optical element.
        Raises:
            NotImplementedError: If the method is not implemented by a
                derived class.
        """
        raise NotImplementedError("intercept() needs to be implemented in derived classes")

    def propagate_ray(self, ray):
        """
        Propagate a ray through, onto, or past the optical element.
        Args:
            ray: Ray to propagate.
        Raises:
            NotImplementedError: If the method is not implemented by a
                derived class.
        """
        raise NotImplementedError("propagate_ray() needs to be implemented in derived classes")

    def plane_intercept(self, ray, z_0, aperture=np.inf):
        """
        Return the intercept of a ray with a plane at a fixed z position.
        The plane is perpendicular to the optical axis and is placed at
        ``z = z_0``. If the ray is parallel to the plane, travelling away from
        it, or misses the aperture, ``None`` is returned.
        Args:
            ray: Ray to intercept with the plane.
            z_0: z-coordinate of the plane.
            aperture: Maximum radial distance accepted by the plane.
        Returns:
            The intercept position, or ``None`` if there is no valid intercept.
        """
        pos = ray.pos()
        direc = ray.direc()
        z_direction = direc[2]

        if z_direction == 0:
            return None

        distance = (z_0 - pos[2]) / z_direction
        if distance <= 0:
            return None

        intercept = pos + distance * direc
        axis_vector = intercept[:2]
        axis_dist = np.linalg.norm(axis_vector)

        if axis_dist > aperture:
            return None

        return intercept

    def focal_point(self):
        """
        Return a numerical estimate of the paraxial focal point.
        Several small ffset rays are propagated through the optical element.
        Their crossings with the optical axis are averaged to estimate the
        focal point. This method is intended for simple focusing elements and
        should not be used for lenses, where the lens makers formula is more
        appropriate.

        Returns:
            The estimated z-coordinate of the focal point.
        """
        ys = [0.01, 0.02, 0.05, 0.1]
        z_crossings = []

        for i in ys:
            paraxial_pos = [0, i, 1]
            paraxial_direc = [0, 0, 1]
            ray = Ray(paraxial_pos, paraxial_direc)
            self.propagate_ray(ray)

            y_direction = ray.direc()[1]
            if y_direction == 0:
                continue

            y_position = ray.pos()[1]
            number_of_vectors = -y_position / y_direction

            if number_of_vectors <= 0:
                continue

            crossing = ray.pos()[2] + number_of_vectors * ray.direc()[2]
            z_crossings.append(crossing)

        return np.mean(z_crossings)


class SphericalSurface(OpticalElement):
    """Represent a spherical optical surface with an aperture."""

    def __init__(
        self,
        *,
        z_0=100.0,
        aperture=34.0,
        curvature=0.03,
        n_1=1.0,
        n_2=1.5,
    ):
        """
        Create a spherical surface.
        The surface vertex is placed at ``z_0`` on the optical axis. A
        curvature of zero creates a plane surface. A non-zero curvature defines
        the radius and centre of the corresponding sphere.
        Args:
            z_0: z-coordinate of the surface vertex.
            aperture: Maximum radial distance accepted by the surface.
            curvature: Curvature of the surface.
            n_1: Refractive index before the surface.
            n_2: Refractive index after the surface.
        """
        self.__z_0 = z_0
        self.__aperture = aperture
        self.__curvature = curvature
        self.__n_1 = n_1
        self.__n_2 = n_2

        if curvature == 0:
            self.__radius = None
            self.__centre = None
        else:
            self.__radius = 1 / curvature
            self.__centre = np.array([0.0, 0.0, self.__z_0 + self.__radius])

    def z_0(self):
        """Return the z-coordinate of the surface vertex."""
        return self.__z_0

    def aperture(self):
        """Return the maximum radial distance accepted by the surface."""
        return self.__aperture

    def curvature(self):
        """Return the curvature of the spherical surface."""
        return self.__curvature

    def n_check(self, n, wavelength):
        """
        Return the rfractive index for a given wavelength.
        If n is a material like object with a ``ref_index`` method, the
        wavelength is used to calculate the refractive index. Otherwise, ``n``
        is treated as a constant refractive index.

        Args:
            n: Constant refractive index or material like object.
            wavelength: Wavelength used to evaluate the refractive index.

        Returns:
            The refractive index.
        """
        if hasattr(n, "ref_index"):
            return n.ref_index(wavelength)
        return n

    def n_1(self, wavelength=588e-6):
        """Return the refractive index before the surface."""
        return self.n_check(self.__n_1, wavelength)

    def n_2(self, wavelength=588e-6):
        """Return the refractive index after the surface."""
        return self.n_check(self.__n_2, wavelength)

    def centre(self):
        """Return the centre of curvature of the spherical surface."""
        return self.__centre

    def intercept(self, ray):
        """
        Return the first valid intercept between a ray and the surface.
        Plane surfaces are handled using ``plane_intercept``. Curved surfaces
        are handled by solving the ray-sphere intersection equation and
        checking that the intercept lies within the aperture.
        Args:
            ray: Ray to intercept with the surface.

        Returns:
            The intercept position, or ``None`` if there is no valid intercept.
        """
        if self.curvature() == 0:
            return self.plane_intercept(ray, self.z_0(), self.aperture())

        pos = ray.pos()
        direc = ray.direc()
        r = pos - self.__centre
        k_hat = direc
        r_dot_k_hat = np.dot(r, k_hat)
        discriminant = r_dot_k_hat**2 - (np.dot(r, r) - self.__radius**2)

        if discriminant < 0:
            return None

        l_1 = -r_dot_k_hat + np.sqrt(discriminant)
        l_2 = -r_dot_k_hat - np.sqrt(discriminant)
        l_vals = np.array([l_1, l_2])

        EPS = 1e-9
        possible_l_vals = l_vals[l_vals > EPS]

        if possible_l_vals.size == 0:
            return None

        intercepts = []

        for i in possible_l_vals:
            intercept = pos + k_hat * i
            axis_vector = intercept[:2]
            axis_dist = np.linalg.norm(axis_vector)

            if axis_dist > self.aperture():
                continue

            intercepts.append(intercept)

        if not intercepts:
            return None

        if len(intercepts) == 1:
            return intercepts[0]

        if self.curvature() > 0:
            intercept = intercepts[np.argmin(possible_l_vals)]
        else:
            intercept = intercepts[np.argmax(possible_l_vals)]

        return intercept


class SphericalRefraction(SphericalSurface):
    """Represent a spherical surface that refracts rays."""

    def propagate_ray(self, ray):
        """
        Propagate a ray by refracting it at the surface.
        The ray is first moved to the intercept point. The surface normal is
        then calculated and passed to the refraction function. If the ray misses
        the surface or total internal reflection occurs, ``None`` is returned.
        Args:
            ray: Ray to refract at the surface.

        Returns:
            The updated ray, or ``None`` if the ray cannot be propagated.
        """
        new_position = self.intercept(ray)

        if new_position is None:
            return None

        direc = ray.direc()

        if self.curvature() != 0:
            normal = new_position - self.centre()
        else:
            normal = np.array([0.0, 0.0, -1.0])

        if np.dot(direc, normal) > 0:
            normal = -normal

        new_direc = physics.refract(
            direc,
            normal,
            self.n_1(ray.wavelength),
            self.n_2(ray.wavelength),
        )

        if new_direc is None:
            return None

        return ray.append(new_position, new_direc)


class Plane(OpticalElement):
    """Represent a plane surface perpendicular to the optical axis."""

    def __init__(self, z_0):
        """
        Create a plane surface at a fixed z position.
        Args:
            z_0: z-coordinate of the plane.
        """
        self.__z_0 = z_0

    def z_0(self):
        """Return the z-coordinate of the plane."""
        return self.__z_0

    def intercept(self, ray):
        """
        Return the intercept between a ray and the plane.
        Args:
            ray: Ray to intercept with the plane.
        Returns:
            The intercept position, or ``None`` if there is no valid intercept.
        """
        intercept = self.plane_intercept(ray, self.z_0())
        return intercept


class OutputPlane(Plane):
    """Represent a plane used to record final ray positions."""

    def propagate_ray(self, ray):
        """
        Propagate a ray to the output plane.
        The ray is moved to its intercept with the plane. Its direction is then
        set along the positive optical axis so that the output plane acts as the
        final recording surface.
        Args:
            ray: Ray to propagate to the output plane.
        Returns:
            The updated ray, or ``None`` if the ray cannot be propagated.
        """
        end_position = self.intercept(ray)

        if end_position is None:
            return None

        direc = [0, 0, 1]
        return ray.append(end_position, direc)


class SphericalReflection(SphericalRefraction):
    """Represent a spherical surface that reflects rays."""

    def propagate_ray(self, ray):
        """
        Propagate a ray by reflecting it at the surface.
        The ray is first moved to the intercept point. The surface normal is
        chosen to face the incoming ray, then the reflected direction is found
        using the reflection function.
        Args:
            ray: Ray to reflect at the surface.
        Returns:
            The updated ray, or ``None`` if the ray cannot be propagated.
        """
        new_position = self.intercept(ray)

        if new_position is None:
            return None

        direc = ray.direc()

        if self.curvature() != 0:
            normal = new_position - self.centre()
        else:
            normal = np.array([0.0, 0.0, -1.0])

        if np.dot(direc, normal) > 0:
            normal = -normal

        new_direc = physics.reflect(direc, normal)
        return ray.append(new_position, new_direc)
