import random
import tensorflow as tf
from keras import backend as K
from replay import *
from state import *
from environment import *
from util import *
from constants import CST
import cv2
from dqn import *
import numpy as np

###############################################################################
#                               Agent Class                                   #
###############################################################################


class Agent:
    def __init__(self, start_p=0, start_s=0):
        """
        Parameters
        ----------
        start_p : float, optional
            Initial position
        start_s : float, optional
            Initial speed
        """
        self.__env = Environment()
        self.__state = State(start_p, start_s)

    def __getNextState(self, action, state=None):
        """
        Compute the next state with the next action from the state given
        in argument. If no state is given, the state of this object is used.

        Parameters
        ----------
        action : {-4, 4}
            Action taken
        state : State Object, optional
          Initial state from which we compute the next one

        Returns
        -------
        state : State Object
            the next state.
            If no state is given as parameters, no return but the self.__state
            object is modified accordindly
        """
        if state is None:
            x = self.__state
        else:
            x = state
        # Formula :
        # yn = yn-1 + h * y'
        # If y = p, y' = s
        # If y = s, y' = formula computed in self.__env.__getDerivativeS
        next_p = x.p + CST.H * x.s
        next_s = x.s + (CST.H * self.__env.DerivativeS(x.p, x.s, action))
        next_state = State(next_p, next_s)

        # Returning if state was given in argument
        if state is None:
            self.__state = next_state
            return
        else:
            return next_state


    def DQN(self, plot=False):
        """
        Implementation of the DQN network. From the images of NB_EPISODES trajs
        estimates the Q function as a tf model.

        Parameters:
        -----------
        plot : boolean, optional
            If set to true, will display plot of the evolution of the agent
        Returns:
        --------
        dqn : DQN object containing the tf model
        """
        BUFFER_SIZE = 100000    # Max size of the history
        BATCH_SIZE = 32         # Size of the mini-batch
        GAMMA = 0.99            # Discount factor
        TAU = 0.01              # Target Network Learning rate

        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        state_dim = (400, 400, 3)
        action_dim = 1

        #neg_action = np.array([-4])
        #pos_action = np.array([4])

        dqn = DQN(sess, state_dim, action_dim, TAU, GAMMA)
        dqn.load_weights()

        current_p = self.__state.p
        current_s = self.__state.s
        state = State(current_p, current_s)

        max_epsilon = 0.8
        epsilon = max_epsilon

        buff = ReplayBuffer(BUFFER_SIZE)

        if plot:
            x_plt = []
            y_plt = []

        save_caronthehill_image(current_p, current_s, "tmpDQN.png")
        first_img = cv2.imread('tmpDQN.png')
        # Will be the intput of the NN
        image_array  = np.array(first_img)

        for n in range(CST.NB_EPISODES):
            print("==========================================================")
            print("Episode {}. Epsilon : {}.".format(n, epsilon))
            total_reward = 0
            while not state.isFinal():
                ################################################################
                # Predicting part
                target_Q_neg = dqn.model.predict([image_array.reshape(1, 400, 400, 3), np.array([-4])])
                target_Q_pos = dqn.model.predict([image_array.reshape(1, 400, 400, 3), np.array([+4])])
                if random.random() < epsilon:
                    action = random.choice([-4, 4])
                elif target_Q_neg > target_Q_pos:
                    action = -4
                else:
                    action = 4
                state = State(current_p, current_s)
                next_state = self.__getNextState(action, state)
                r = next_state.getReward()
                total_reward += r

                current_p = next_state.p
                current_s = next_state.s
                state = next_state

                save_caronthehill_image(current_p, current_s, "tmpDQN.png")
                new_img = cv2.imread('tmpDQN.png')
                # Will be the intput of the NN
                new_image_array  = np.array(new_img)

                buff.add(image_array, action, r, new_image_array)
                image_array = new_image_array
                ################################################################
                # Training part
                # Get experiences from the batch
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])

                y_t = np.asarray([e[1] for e in batch])
                # Compute the value of the Q from the dqn
                target_q_values = dqn.target_model.predict([new_states, actions])
                # Compute the true y
                for k in range(len(batch)):
                    y_t[k] = rewards[k] + CST.DISCOUT_FACTOR*target_q_values[k]

                # Update the dqn
                current_loss = dqn.model.train_on_batch([states,actions], y_t)
                dqn.target_train()
                dqn.save_weights()
            print("Total reward for the episode: {}.".format(total_reward))
            if plot:
                x_plt.append(n)
                y_plt.append(total_reward)
                # Printing the graph at each episode for 2 reasons :
                # 1) See the evolution while running
                # 2) Still having graphs if we CTRL+C the execution
                simplePlot(x_plt, y_plt, "Episodes", "Total Reward", "DQN")
            epsilon = max_epsilon - (n/CST.NB_EPISODES)
            dqn.save_weights(True)
        if plot:
            simplePlot(x_plt, y_plt, "Episodes", "Total Reward", "DQN")
        os.remove("tmpDQN.png")
        return dqn
