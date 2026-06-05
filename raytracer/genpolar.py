"""Generate concentric points"""
import numpy as np

def rtrings(rmax, nrings, multi):
    """Returns points on concentric rings"""
    yield [0, 0, 0]
    for i in range(1, nrings + 1):
        radius = (rmax / nrings) * i
        n_points = multi * i
        for j in range(n_points):
            theta = 2 * np.pi * j / n_points
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            yield [x, y, 0]
