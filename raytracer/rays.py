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
            pos = [0, 0, 0]

        if direc is None:
            direc = [0, 0, 1]

        self.__pos = [np.array(pos)]
        self.__direc = [np.array(direc)]

    def pos(self):
        """Return the latest position of the ray"""
        return self.__pos[-1].copy()

    def direc(self):
        """Return the latest direction of the ray"""
        return self.__direc[-1].copy()

    def append(self, new_pos, new_direc):
        """Add a new position and direction to the ray"""
        self.__pos.append(np.array(new_pos))
        self.__direc.append(np.array(new_direc))

    def vertices(self):
        """Return the direction history of the ray """
        return self.__pos
