"""Represents the optical system using optical elements such as refracting surfaces, output plane""" 

import numpy as np
from raytracer import physics
from raytracer.rays import Ray

class OpticalElement:
    "Base class for all Optical elements"
    def intercept(self, ray):
        """Base for the intercept"""
        raise NotImplementedError('intercept() needs to be implemented in derived classes')

    def propagate_ray(self, ray):
        """Base for the propogate method"""
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')

    def plane_intercept(self, ray, z_0, aperture = np.inf):
        """Return intercept with plane z = z_0."""
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
        """Calculates the paraxial focus of this object"""
        #ys = np.linspace(0.01, 0.1, 50)
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

class SphericalRefraction(OpticalElement):
    """Shperical refraction implementation"""
    def __init__(
        self,
        *,
        z_0 = 100.0,
        aperture = 34.0,
        curvature = 0.03,
        n_1 = 1.0,
        n_2 = 1.5
    ):
        """Create spherical refraction object"""
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
        """Returns a copy of z0"""
        return self.__z_0

    def aperture(self):
        """Returns a copy of aperture"""
        return self.__aperture

    def curvature(self):
        """Returns a copy of curvature"""
        return self.__curvature

    def n_1(self):
        """Returns a copy of the n_1 value"""
        return self.__n_1

    def n_2(self):
        """Returns a copy of the n_2 value"""
        return self.__n_2

    def centre(self):
        """Returns the centre"""
        return self.__centre

    def intercept(self, ray):
        """Return the closest valid ray intercept with the spherical surface."""
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
        possible_l_vals = l_vals[l_vals > 0]

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

    def propagate_ray(self, ray):
        """Propoagates ray"""
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

        new_direc = physics.refract(direc, normal, self.n_1(), self.n_2())

        return ray.append(new_position, new_direc)

class OutputPlane(OpticalElement):
    """Plane"""
    def __init__(self, z_0):
        """Create Plane"""
        self.__z_0 = z_0

    def z_0(self):
        """Return the z_0"""
        return self.__z_0

    def intercept(self, ray):
        """Intercept"""
        intercept = self.plane_intercept(ray, self.z_0())
        return intercept

    def propagate_ray(self, ray):
        """Propogates ray"""
        end_position = self.intercept(ray)
        if end_position is None:
            return None
        direc = [0, 0, 1]
        ray.append(end_position, direc)
