"""Ray and ray-bundle classes for simple ray tracing."""

import numpy as np
import matplotlib.pyplot as plt
from raytracer.genpolar import rtrings


class Ray:
    """Represent a single ray travelling through 3D space."""
    def __init__(self, pos=None, direc=None, wavelength=588e-6):
        """
        Creates a ray with a position, direction, and wavelength.
        If no position or direction is given, the ray starts at the origin
        and travels along the positive z-axis.

        Args:
            pos: Initial 3D position of the ray.
            direc: Initial 3D direction of the ray.
            wavelength: Wavelength of the ray.
        """
        if pos is None:
            pos = np.array([0, 0, 0], dtype=float)
        if direc is None:
            direc = np.array([0, 0, 1], dtype=float)

        pos = np.array(pos, dtype=float)
        direc = np.array(direc, dtype=float)
        Ray.dimension_check(self, pos, direc)
        Ray.normalise(self, direc)

        self.__pos = [pos]
        self.__direc = direc
        self.__wavelength = wavelength

    def normalise(self, vector):
        """
        Scale a vector so that it has unit length.
        Args:
            vector: Vector to normalise.
        Returns:
            The normalised vector.
        Raises:
            ValueError: If the vector has zero magnitude.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("The magnitude of the direction vector is 0")
        vector[:] = vector / norm
        return vector

    @property
    def wavelength(self):
        """Return the wavelength of the ray."""
        return self.__wavelength

    def dimension_check(self, position, direction):
        """
        Check that the position and direction are both 3D vectors.
        Args:
            position: Position vector to check.
            direction: Direction vector to check.
        Raises:
            TypeError: If either vector is not three-dimensional.
        """
        if len(direction) < 3:
            raise TypeError("The direction inserted is less than 3D")
        if len(direction) > 3:
            raise TypeError("The direction inserted is greater than 3D")
        if len(position) < 3:
            raise TypeError("The position inserted is less than 3D")
        if len(position) > 3:
            raise TypeError("The position inserted is greater than 3D")

    def pos(self):
        """Return a copy of the ray's current position."""
        return self.__pos[-1].copy()

    def direc(self):
        """Return the ray's current direction."""
        return self.__direc

    def append(self, pos, direc):
        """Add a new point and direction to the ray path.
        Args:
            pos: New 3D position of the ray.
            direc: New 3D direction of the ray.
        Returns:
            The updated ray.
        """
        self.dimension_check(pos, direc)
        self.normalise(direc)
        self.__pos.append(np.array(pos))
        self.__direc = np.array(direc)
        return self

    def vertices(self):
        """Return all positions visited by the ray."""
        return self.__pos

    @property
    def z_int(self):
        """
        Return the z-coordinate where the ray crosses the optical axis.

        The optical axis is taken to be the z-axis, where x and y are both
        zero.

        Returns:
            The z-coordinate of the intercept.

        Raises:
            ValueError: If the ray does not intercept the optical axis.
        """
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
    """Represent a group of rays with starting positions in rings."""

    def __init__(self, rmax=5.0, nrings=5, multi=6):
        """Create a bundle of rays arranged in concentric rings.

        Args:
            rmax: Maximum radius of the ray bundle.
            nrings: Number of rings in the bundle.
            multi: Number of rays added per ring multiplier.
        """
        self.__positions_3d = np.array(list(rtrings(rmax, nrings, multi)))
        self.__rays = list(self.ray_bundle(self.__positions_3d))

    @property
    def rays(self):
        """Return the rays in the bundle."""
        return self.__rays

    @rays.setter
    def rays(self, value):
        """Set the rays in the bundle."""
        self.__rays = value

    def ray_bundle(self, positions):
        """
        Generate rays from a set of starting positions.
        Each ray starts at one of the given positions and initially travels
        along the positive z-axis.
        Args:
            positions: Iterable of 3D starting positions.
        Yields:
            Ray objects starting from the given positions.
        """
        direc = [0, 0, 1]

        for position in positions:
            ray = Ray(position, direc)
            yield ray

    def propagate_bundle(self, elements):
        """
        Propagate all rays through a sequence of optical elements.
        Rays that are not successfully propagated are removed from the bundle.
        Args:
            elements: Optical elements with a ``propagate_ray`` method.
        """
        for element in elements:
            surviving_rays = []

            for ray in self.rays:
                result = element.propagate_ray(ray)
                if result is not None:
                    surviving_rays.append(ray)

            self.rays = surviving_rays

    def track_plot(self):
        """
        Plot the 3D paths followed by the rays.
        Returns:
            The matplotlib figure containing the ray paths.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.set_xlabel("x / mm")
        ax.set_ylabel("y / mm")
        ax.set_zlabel("z / mm")

        for ray in self.rays:
            vertices = np.array(ray.vertices())

            z_values = vertices[:, 2]
            y_values = vertices[:, 1]
            x_values = vertices[:, 0]

            ax.plot(x_values, y_values, z_values)

        return fig

    def xy(self, rays):
        """
        Return the current x-y positions of a set of rays.
        Args:
            rays: Rays whose current positions should be read.
        Returns:
            A NumPy array containing the current x-y positions.
        """
        x_y_vertices = []

        for ray in rays:
            x_y_vertex = ray.pos()[:2]
            x_y_vertices.append(x_y_vertex)

        return np.array(x_y_vertices)

    def rms(self):
        """Return the RMS spot size of the current ray positions."""
        x_y_vertices = self.xy(self.rays)
        magnitude_squared = np.sum(x_y_vertices ** 2, axis=1)
        rms = np.sqrt(np.mean(magnitude_squared))

        return rms

    def spot_plot(self, fig=None, label=None):
        """
        Plot the current x-y ray positions as a spot diagram.
        Args:
            fig: Existing matplotlib figure to plot on. If not given, a new
                figure is created.
            label: Optional label for the plotted points.

        Returns:
            The matplotlib figure containing the spot plot.
        """
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
