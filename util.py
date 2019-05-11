import cv2
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from constants import CST
from displaycaronthehill import save_caronthehill_image

###############################################################################
#                               Utils methods                                 #
###############################################################################


def createVideo(states, name):
    """
    Create a mp4 video of given name from a list of states given in
    argument. This method uses opencv.

    Parameters
    ----------
    states : list[State]
        List of the states to display
    name : str
        Name of the video to create
    """
    # Creating the first image to get the size of the image
    save_caronthehill_image(states[0].p, states[0].s, "tmpCOTH.png")
    frame = cv2.imread("tmpCOTH.png", 1)
    height, width, layers = frame.shape
    size = (width, height)

    # Opencv settings
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(name + '.avi', fourcc, CST.FPS, size)

    print("""Creating video file of the simulation """
          """({} frames)""".format(len(states)))
    # Only considerating 1 over CST.FRAME_STEP frame to gain performance
    for i in range(0, len(states), CST.FRAME_STEP):
        print("Currently working on frame {}".format(i), end='\r')

        # Creating the video from image and deleting temp image
        save_caronthehill_image(states[i].p, states[i].s, "tmpCOTH.png")
        frame = cv2.imread("tmpCOTH.png", 1)
        out.write(frame)
        os.remove("tmpCOTH.png")
    # Video creation
    out.release()
    print("\nVideo complete!")
    cv2.destroyAllWindows()


def plot3D(list):
    """
    Creating a 3D plot from the list of J.
    """
    # Getting X and Y
    y = np.linspace(-1, 1, 21)
    x = np.linspace(-3, 3, 61)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    # Creating Z from the list
    l = len(Z)
    w = len(Z[0])
    i = 0
    while i < l:
        j = 0
        while j < w:
            Z[i][j] = list[i][j]
            j = j + 1
        i = i + 1

    # Plot itself
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel("Speed")
    ax.set_ylabel("Position")
    ax.set_zlabel("Expected return")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis',
                    edgecolor='none')

    plt.show()


def DQNSimulation(dqn, start_state, video_indc=False):
    """
    Runs a simulation based on the DQN object created whle training the agent
    on the image directly.
    We give it 1000 frames to finish. Otherwise, it is considered as stuck and
    thus failing.

    Parameters:
    -----------
    dqn : dqn object
    start_state: State object
        Starting state of the agent
    video : boolean, optional
        If set true, will create a video of the agent
    """
    save_caronthehill_image(current_p, current_s, "tmpDQN.png")
    first_img = cv2.imread('tmpDQN.png')
    # Will be the intput of the NN
    image_array  = np.array(first_img)

    current_p = start_state.p
    current_s = start_state.s
    state = start_state

    if video_indc:
        states_to_display = []
        states_to_display.append(state)

    for i in range(CST.MAX_NB_FRAME):
        # Predicting
        target_Q_neg = dqn.model.predict([image_array.reshape(1, 400, 400, 3), np.array([-4])])
        target_Q_pos = dqn.model.predict([image_array.reshape(1, 400, 400, 3), np.array([+4])])
        if target_Q_neg > target_Q_pos:
            action = -4
        else:
            action = 4

        next_state = self.__getNextState(action, state)
        if video_indc:
            states_to_display.append(next_state)
        current_p = next_state.p
        current_s = next_state.s
        state = next_state

        save_caronthehill_image(current_p, current_s, "tmpDQN.png")
        new_img = cv2.imread('tmpDQN.png')
        # Will be the intput of the NN
        image_array  = np.array(new_img)
    os.remove("tmpDQN.png")

    if video_indc:
        name = "DQN_agent"
        createVideo(states_to_display, name)

def simplePlot(x, y, name_x, name_y, file_name):
    """
    Create a simple evolution plot of the given array.

    Parameters:
    -----------
    x : []
        Containing all the abcssica values
    y : []
        Containing all the y values
    name_x : str
        Name of the x axis
    name_y : str
        Name of the y axis
    file_name : str
        Name of the file the function will safe.
    """
    fig = plt.figure()
    print("Making the plots!")
    plt.plot(x, y)
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.savefig(file_name + '.png')
