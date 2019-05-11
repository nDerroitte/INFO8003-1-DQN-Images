import numpy as np
import math
from keras.layers import Dense, Input, Concatenate, Conv2D, Lambda, Flatten
from keras.models import  Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class DQN(object):
    def __init__(self, sess, state_dim, action_dim, TAU, GAMMA):
        """
        Parameters:
        -----------
        sess : A tensorflow session
        state_dim : int
            Number of dimension of the state-space
        action_dim : int
            Number of dimension of the action-space
        TAU : double
            Learning rate of the target network
        GAMMA: double
            Learning rate of the AdamOptimizer
        """
        self.sess = sess
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.state_dim = state_dim
        self.action_dim = action_dim

        K.set_session(sess)

        self.model = self.__create_model()
        self.target_model= self.__create_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def target_train(self):
        """
        Simple call of the predict function of the keras model for the target
        network
        Parameters:
        -----------
        states : []
            List of the 4 elements composing the states
        """
        weights_model = self.model.get_weights()
        weights_target = self.target_model.get_weights()
        for i in range(len(weights_model)):
            weights_target[i] = self.TAU * weights_model[i] + (
                                       (1 - self.TAU) * (weights_target[i]))
        self.target_model.set_weights(weights_target)

    def __create_model(self):
        """
        Create the actor model

        Returns:
        ---------
        keras Model : the actor network
        Action Input
        State Input
        """
        A = Input(shape =[self.action_dim])
        a1 = Dense(256, activation='linear')(A)

        IMG = Input(shape=self.state_dim)
        normalized = Lambda(lambda x: x / 255.0)(IMG)
        l1 = Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)
        l2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(l1)
        flatten  = Flatten()(l2)
        l3 = Dense(128, activation='relu')(flatten)

        merge = Concatenate(axis=-1)([flatten, a1])

        output = Dense(1,activation='relu')(merge)

        model = Model(inputs=[IMG, A],outputs=output)
        adam = Adam(lr=self.GAMMA)
        model.compile(loss='mse', optimizer=adam)
        return model


    def load_weights(self):
        """
        Load the weights of the network if they exist
        """
        try:
            self.model.load_weights('dqn_w.h5')
        except:
            print("No 'dqn_w.h5' file found, using new weights.")

    def save_weights(self, safe=False):
        """
        Safe the weights of the network

        Parameters:
        ----------
        safe : boolean, optional
            If true, safe the network under an other name/
        """
        if safe:
            self.model.save_weights('dqn_wS.h5')
        else:
            self.model.save_weights('dqn_w.h5')
