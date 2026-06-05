"""Physics functions used by the ray tracer."""

import numpy as np

def angle(a, b):
    """
    Return the angle between two vectors.
    Args:
        a: First vector.
        b: Second vector.
    Returns:
        The angle between the vectors in radian
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def normalize(direc, normal):
    """
    Return unit direction and normal vectors.
    Args:
        direc: Direction vector to normalise.
        normal: Surface normal vector to normalise.
    Returns:
        A tuple containing the normalised direction and normal vectors.
    """
    direc = np.array(direc, dtype=np.float64)
    normal = np.array(normal, dtype=np.float64)
    direc_hat = direc / np.linalg.norm(direc)
    normal_hat = normal / np.linalg.norm(normal)
    return direc_hat, normal_hat


def refract(direc, normal, n_1, n_2):
    """
    Return the refracted ray direction at a surface.
    The ray is refracted using Snell's law. If total internal reflection
    occurs, no refracted ray is returned.
    Args:
        direc: Incident ray direction.
        normal: Surface normal at the point of refraction.
        n_1: Refractive index of the incident medium.
        n_2: Refractive index of the transmitted medium.

    Returns:
        The refracted ray direction, or None if total internal reflection
        occurs.
    """
    direc_hat, normal_hat = normalize(direc, normal)
    n_ratio = n_1 / n_2
    theta_i = angle(direc, -normal_hat)
    sin_theta_r = n_ratio * np.sin(theta_i)

    if abs(sin_theta_r) > 1:
        return None

    theta_r = np.arcsin(sin_theta_r)

    if np.isclose(theta_i, 0.0):
        return -normal_hat

    u = np.sin(theta_r)
    v = -np.cos(theta_r)
    direc_new_basis = np.array([u, v])

    tangent = direc_hat - np.dot(direc_hat, normal_hat) * normal_hat
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm == 0:
        return direc_hat

    tangent_hat = tangent / tangent_norm

    basis_2d = np.column_stack((tangent_hat, normal_hat))
    new_direc_r = basis_2d @ direc_new_basis

    return new_direc_r


def reflect(direc, normal):
    """
    Return the reflected ray direction at a surface.
    Args:
        direc: Incident ray direction.
        normal: Surface normal at the point of reflection.
    Returns:
        The reflected ray direction as a unit vector.
    """
    direc_hat, normal_hat = normalize(direc, normal)
    new_direc_refl = direc_hat - (2 * (np.dot(direc_hat, normal_hat) * normal_hat))
    return new_direc_refl / np.linalg.norm(new_direc_refl)


class DispersiveMaterial:
    """Represent a material with Sellmeier dispersion."""

    def __init__(self, b_coeff=(1., 2., 3.), c_coeff=(4., 5., 6.)):
        """
        Create a dispersive material from Sellmeier coefficients.
        Args:
            b_coeff: Dimensionless Sellmeier B coefficients.
            c_coeff: Sellmeier C coefficients in mm squared
        """
        self.__b_coeff = b_coeff
        self.__c_coeff = c_coeff

    def b_coeff(self):
        """Return the Sellmeier B coefficients."""
        return self.__b_coeff

    def c_coeff(self):
        """Return the Sellmeier C coefficients"""
        return self.__c_coeff

    def ref_index(self, wavelength):
        """
        Return the refractive index at a given wavelength.
        Args:
            wavelength: Wavelength in mm."""

        wavelength_squared = wavelength ** 2
        total = 1.0

        for b_coeff, c_coeff in zip(self.b_coeff(), self.c_coeff()):
            total += (b_coeff * wavelength_squared) / (
                wavelength_squared - c_coeff
            )

        return np.sqrt(total)
