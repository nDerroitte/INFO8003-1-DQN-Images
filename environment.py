from constants import CST
import cv2
import os
from displaycaronthehill import save_caronthehill_image

###############################################################################
#                               Agent Class                                   #
###############################################################################


class Environment:
    def __init__(self):
        self.nb_p = int((2 * CST.MAX_POS) / CST.POS_DISCRETISATION_STEP)
        self.nb_s = int((2 * CST.MAX_SPEED) / CST.SPEED_DISCRETISATION_STEP)

    def DerivativeS(self, p, s, action):
        """
        Compute the derivative of s based on the current position and the
        action taken

        Parameters
        ----------
        p : float [-1, 1]
            Position
        s : float [-3, 3]
            Speed
        action : {-4, 4}
            Action taken

        Returns
        -------
        float
            The derivative of s based on the current position and the action
            taken
        """
        hill_derivative = self.__get_hill_derivative(p)
        hill_second_derivative = self.__get_hill_second_derivative(p)

        # Computing term one by one
        term1 = action / (CST.M * (1 + pow(hill_derivative, 2)))
        term2 = CST.G * hill_derivative / (1 + pow(hill_derivative, 2))
        term3 = pow(s, 2) * hill_derivative * hill_second_derivative / (
                1 + pow(hill_derivative, 2))
        return term1 - term2 - term3

    def __get_hill_derivative(self, p):
        """
        Compute the first derivative of Hill(p)

        Parameters
        ----------
        p : float [-1, 1]
            Position

        Returns
        -------
        float
            First derivative of Hill(p) : Hill'(p)
        """
        if p < 0:
            return 2 * p + 1
        else:
            return 1 / pow(5 * pow(p, 2) + 1, 3 / 2)

    def __get_hill_second_derivative(self, p):
        """
        Compute the second derivative of Hill(p)

        Parameters
        ----------
        p : float [-1, 1]
            Position

        Returns
        -------
        float
            Second derivative of Hill(p) : Hill''(p)
        """
        if p < 0:
            return 2
        else:
            return -15 * p / pow(5 * pow(p, 2) + 1, 5 / 2)

    def pos_to_index(self, pos):
        """
        Transform the position [-1, 1] to an index [0, 20] by time step of 0.1

        Parameters
        ----------
        pos : float [-1, 1]
            Position

        Returns
        -------
        int
            Corresponding index of the position
        """

        return round((pos+CST.MAX_POS)/CST.POS_DISCRETISATION_STEP)

    def speed_to_index(self, speed):
        """
        Transform the speed [-3, 3] to an index [0, 60] by time step of 0.1

        Parameters
        ----------
        speed : float [-3, 3]
            Speed

        Returns
        -------
        int
            Corresponding index of the speed
        """
        return round((speed+CST.MAX_SPEED)/CST.SPEED_DISCRETISATION_STEP)

    def index_to_pos(self, index):
        """
        Transform the position index [0, 20] to real position [-1, 1]

        Parameters
        ----------
        index : int
            Index of the position

        Returns
        -------
        float
            Corresponding real position
        """
        return (index * CST.POS_DISCRETISATION_STEP) - CST.MAX_POS

    def index_to_speed(self, index):
        """
        Transform the speed index [0, 60] to real position [-3, 3]

        Parameters
        ----------
        index : int
            Index of the speed

        Returns
        -------
        float
            Corresponding real speed
        """
        return (index * CST.SPEED_DISCRETISATION_STEP) - CST.MAX_SPEED
