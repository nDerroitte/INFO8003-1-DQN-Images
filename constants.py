###############################################################################
#                               CST Class                                     #
###############################################################################


class CST:
    ###########################################################################
    #                   Constants that won't change ever                      #
    ###########################################################################
    # Physic related
    G = 9.81
    M = 1
    MAX_POS = 1
    MAX_SPEED = 3
    BOUND_U = 4
    # Direction related
    DIR_LEFT = -4
    DIR_RIGHT = 4
    DIR_RANDOM = 0
    NB_ACTION = 2
    # Algorithm related
    NB_EPISODES_MONTE_CARLO = 1 # deterministic problem
    RANDOM_SEED = 77
    ###########################################################################
    #  Constants that may be changed for studying the behaviour of the agent  #
    ###########################################################################
    # Algo related
    ALPHA = 0.05  # Q Learning algorithm
    # Discretization constants
    H = 0.001  # Integration step
    POS_DISCRETISATION_STEP = 0.1
    SPEED_DISCRETISATION_STEP = 0.1
    TIME_DISCRETISATION_STEP = 0.1  # In seconds. 10 states per sec
    # Video related variables
    # Above 900 frames, opencv makes Pygame bugs on IOS.
    # Pygame was not made for IOS and it shows.
    # That's why, the number of frames can not be above 36s
    FPS = 25.0  # Fps for video. =/= nb of discrete time.
    # /!\ Time in video =/= t in statement by a factor FPS.

    # Only showing 1 out of 10 frames to gain comp. time and fluidity
    FRAME_STEP = 10
    ###########################################################################
    #               Constants that are parameters of the problem              #
    ###########################################################################
    # Default value. Will be reset when executing the main in the run.py
    DISCOUT_FACTOR = 0.95
    TIME_VISUALISATION = 10  # In seconds
    NB_EPISODES = 100
    LENGTH_EPISODE = 10000
    ###########################################################################
    #           Computed constants : depends on other constants               #
    ###########################################################################
    MAX_NB_FRAME = int(TIME_VISUALISATION * FPS / TIME_DISCRETISATION_STEP)
