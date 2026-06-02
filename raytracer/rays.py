"""Rays and their properties"""

import numpy as np

class Ray:
    """Ray object"""

    def __init__(self, pos = None, direc = None):
        """
        Initialize and creates a ray object which has a origin position
        and a direction. 
        """

        if pos is None:
            pos = np.array([0, 0, 0], dtype = float)

        if direc is None:
            direc = np.array([0, 0, 1], dtype = float)

        pos = np.array(pos, dtype = float)
        direc = np.array(direc, dtype = float)
        self.dimension_check(pos, direc)
        self.normalise(direc)
        self.__pos = [pos]
        self.__direc = direc


    def normalise(self, vector):
        """Normalizes a vector and alters it"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("The magnitude of the direction vector is 0")
        vector[:] = vector/norm
        return vector


    def dimension_check(self, position, direction):
        """Checks dimensions of input arrays"""

        if len(direction) < 3:
            raise TypeError("The direction inserted is less than 3D")
        if len(direction) > 3:
            raise TypeError("The direction inserted is greater than 3D")
        if len(position) < 3:
            raise TypeError("The position inserted is less than 3D")
        if len(position) > 3:
            raise TypeError("The position inserted is greater than 3D")

    def pos(self):
        """Return the latest position of the ray"""
        return self.__pos[-1].copy()

    def direc(self):
        """Return the latest direction of the ray"""
        return self.__direc

    def append(self, pos, direc):
        """Add a new position and direction to the ray"""
        self.dimension_check(pos, direc)
        self.normalise(direc)
        self.__pos.append(np.array(pos))
        self.__direc = np.array(direc)

    def vertices(self):
        """Return the position history of the ray """
        return self.__pos
