"""Optical elements""" 

import numpy as np
from raytracer import physics
from raytracer.rays import Ray

class OpticalElement:
    "Base optical element"
    def intercept(self, ray):
        """Find intercept"""
        raise NotImplementedError('intercept() needs to be implemented in derived classes')

    def propagate_ray(self, ray):
        """Propagate ray"""
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')

    # All optical elements need to be flexible enough to handle plane intercepts, thus defined here
    def plane_intercept(self, ray, z_0, aperture = np.inf):
        """Return plane intercept"""
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

    # Numerical approach to find the focus. WILL NOT WORK FOR LENSES - Use Lens Makers Formula
    def focal_point(self):
        """Return paraxial focus" found via ray tracing"""
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
    """Spherical surface"""
    def __init__(
        self,
        *,
        z_0 = 100.0,
        aperture = 34.0,
        curvature = 0.03,
        n_1 = 1.0,
        n_2 = 1.5
    ):
        """Make spherical surface"""
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
        """Return z0"""
        return self.__z_0

    def aperture(self):
        """Return aperture"""
        return self.__aperture

    def curvature(self):
        """Return curvature"""
        return self.__curvature

    def n_check(self, n, wavelength):
        """Return refractive index"""
        if hasattr(n, "ref_index"):
            return n.ref_index(wavelength)
        return n

    def n_1(self, wavelength = 588e-6):
        """Return n1"""
        return self.n_check(self.__n_1, wavelength)

    def n_2(self, wavelength = 588e-6):
        """Return n2"""
        return self.n_check(self.__n_2, wavelength)

    def centre(self):
        """Return centre"""
        return self.__centre

    def intercept(self, ray):
        """Return spherical intercept"""
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
    """Spherical refraction"""
    def propagate_ray(self, ray):
        """Propagate ray"""
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

        new_direc = physics.refract(direc, normal, self.n_1(ray.wavelength), self.n_2(ray.wavelength))

        if new_direc is None:
            return None

        return ray.append(new_position, new_direc)

class Plane(OpticalElement):
    """Plane surface"""
    def __init__(self, z_0):
        """Make plane"""
        self.__z_0 = z_0

    def z_0(self):
        """Return z0"""
        return self.__z_0

    def intercept(self, ray):
        """Return intercept"""
        intercept = self.plane_intercept(ray, self.z_0())
        return intercept

class OutputPlane(Plane):
    """Output plane"""
    def propagate_ray(self, ray):
        """Propagate ray"""
        end_position = self.intercept(ray)
        if end_position is None:
            return None
        direc = [0, 0, 1]
        return ray.append(end_position, direc)


class SphericalReflection(SphericalRefraction):
    """Spherical reflection"""
    def propagate_ray(self, ray):
        """Reflect ray"""
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
