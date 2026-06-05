"""Physics module that implements refractions at incidence points"""

import numpy as np

def angle(a, b):
    """Calculates the angle between two position vectors"""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def normalize(direc, normal):
    """Normalization helper"""
    direc = np.array(direc, dtype=np.float64)
    normal = np.array(normal, dtype=np.float64)
    direc_hat = direc / np.linalg.norm(direc)
    normal_hat = normal / np.linalg.norm(normal)
    return direc_hat, normal_hat


def refract(direc, normal, n_1, n_2):
    """Calculate the new refracted ray direction"""
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
    """Reflection function"""
    direc_hat, normal_hat = normalize(direc, normal)
    new_direc_refl = direc_hat - (2 * (np.dot(direc_hat, normal_hat) * normal_hat))
    return new_direc_refl / np.linalg.norm(new_direc_refl)