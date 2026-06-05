"""Ray and ray bundle classes"""

import numpy as np
import matplotlib.pyplot as plt
from raytracer.genpolar import rtrings

class Ray:
    """Single ray"""
    def __init__(self, pos = None, direc = None, wavelength = 588e-6):
        """Initializes ray"""
        if pos is None:
            pos = np.array([0, 0, 0], dtype = float)
        if direc is None:
            direc = np.array([0, 0, 1], dtype = float)
        pos = np.array(pos, dtype = float)
        direc = np.array(direc, dtype = float)
        Ray.dimension_check(self, pos, direc)
        Ray.normalise(self, direc)
        self.__pos = [pos]
        self.__direc = direc
        self.__wavelength = wavelength


    def normalise(self, vector):
        """Normalizes vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("The magnitude of the direction vector is 0")
        vector[:] = vector/norm
        return vector

    @property
    def wavelength(self):
        """Returns wavelength"""
        return self.__wavelength

    def dimension_check(self, position, direction):
        """Checks vector size"""
        if len(direction) < 3:
            raise TypeError("The direction inserted is less than 3D")
        if len(direction) > 3:
            raise TypeError("The direction inserted is greater than 3D")
        if len(position) < 3:
            raise TypeError("The position inserted is less than 3D")
        if len(position) > 3:
            raise TypeError("The position inserted is greater than 3D")

    def pos(self):
        """Returns current position"""
        return self.__pos[-1].copy()

    def direc(self):
        """Returns current direction"""
        return self.__direc

    def append(self, pos, direc):
        """Adds point to ray"""
        self.dimension_check(pos, direc)
        self.normalise(direc)
        self.__pos.append(np.array(pos))
        self.__direc = np.array(direc)
        return self

    def vertices(self):
        """Returns ray points"""
        return self.__pos

    @property
    def z_int(self):
        """Returns optical axis z intercept"""
        pos = self.pos()
        direc = self.direc()
        x_pos = pos[0]
        y_pos = pos[1]
        x_direc = direc[0]
        y_direc = direc[1]

        if not np.isclose(x_direc, 0.0):
            number_vectors = -x_pos / x_direc

        elif not np.isclose(y_direc, 0.0):
            number_vectors = -y_pos / y_direc

        else:
            if np.allclose(pos[:2], [0.0, 0.0]):
                return pos[2]
            raise ValueError("The ray does not intercept the optical axis.")

        if number_vectors < 0:
            raise ValueError("The ray does not intercept the optical axis.")

        intercept_position = pos + number_vectors * direc

        if not np.allclose(intercept_position[:2], [0.0, 0.0]):
            raise ValueError("The ray does not intercept the optical axis.")

        return intercept_position[2]

class RayBundle:
    """Ray bundle"""
    def __init__(self, rmax = 5.0, nrings = 5, multi = 6):
        """Initializes ray bundle"""
        self.__positions_3d = np.array(list(rtrings(rmax, nrings, multi)))
        self.__rays = list(self.ray_bundle(self.__positions_3d))

    @property
    def rays(self):
        """Returns rays"""
        return self.__rays

    @rays.setter
    def rays(self, value):
        """Sets rays"""
        self.__rays = value

    def ray_bundle(self, positions):
        """Makes rays from positions"""
        direc = [0, 0, 1]
        for i in positions:
            ray = Ray(i, direc)
            yield ray

    def propagate_bundle(self, elements):
        """Propagates rays"""
        for element in elements:
            surviving_rays = []
            for ray in self.rays:
                result = element.propagate_ray(ray)
                if result is not None:
                    surviving_rays.append(ray)
            self.rays = surviving_rays

    def track_plot(self):
        """Plots ray paths"""
        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        for ray in self.rays:
            vertices = np.array(ray.vertices())
            z_values = vertices[:, 2]
            y_values = vertices[:, 1]
            x_values = vertices[:, 0]
            ax.plot(x_values, y_values, z_values)

        return fig

    def xy(self, rays):
        """Returns xy positions"""
        x_y_vertices = []
        for ray in rays:
            x_y_vertex = ray.pos()[:2]
            x_y_vertices.append(x_y_vertex)
        return np.array(x_y_vertices)

    def rms(self):
        """Returns rms spot size"""
        x_y_vertices = self.xy(self.rays)
        magnitude_squared = np.sum(x_y_vertices ** 2, axis = 1)
        rms = np.sqrt(np.mean(magnitude_squared))
        return rms

    def spot_plot(self, fig = None, label = None):
        """Plots ray spots"""
        x_y_vertices = self.xy(self.rays)
        x_position = x_y_vertices[:, 0]
        y_position = x_y_vertices[:, 1]

        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)

        plt.scatter(x_position, y_position, s=10, label=label)
        plt.axis("equal")
        plt.xlabel("x position at output plane (mm)")
        plt.ylabel("y position at output plane (mm)")
        plt.grid(True)

        if label is not None:
            plt.legend()
        return fig
