from argparse import ArgumentParser, ArgumentTypeError
from constants import CST
from agent import *
import time
from util import *

if __name__ == "__main__":
    start = time.time()
    usage = """
    USAGE:      python3 run.py <options>
    EXAMPLES:   (1) python run.py
                    - Launch the training of the DQN.
    """

    # Using argparse to select the different setting for the run
    parser = ArgumentParser(usage)

    # nb_episodes
    parser.add_argument(
        '--nb_episodes',
        help='Number of episodes used during the Q-learing algorithm',
        type=int,
        default=200
    )

    # discount_factor : gamma parameter
    parser.add_argument(
        '--discount_factor',
        help='Discount factor (gamma)',
        type=float,
        default=0.99
    )

    # video
    parser.add_argument(
        '--video',
        help="""1 if the user wants a video of the simulation. 0 otherwise.""",
        type=int,
        default=0
    )

    # plot
    parser.add_argument(
        '--plot',
        help="""1 if the user wants plot of the simulation. 0 otherwise.""",
        type=int,
        default=0
    )

    args = parser.parse_args()
    video_indc = args.video
    plot_indc = args.plot
    CST.NB_EPISODES = args.nb_episodes
    CST.DISCOUT_FACTOR = args.discount_factor

    # Implementing a DQN
    a = Agent(start_p=0)
    # Training the DQN
    dqn = a.DQN(plot_indc)
    # Can't use a.updatePolicy since it only works for non-visuale cases
    # Use of a new function running the simulation
    DQNSimulation(dqn, State(0, 0), video_indic=video_indc)
    # Can't use a.evaluatePolicy nether.

    print("--------- Comp. time : {} ---------".format(time.time() - start))
