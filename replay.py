from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        """
        Parameters:
        -----------
        buffer_size : int
            Size of the history
        """
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        """
        Get a random mini-batch of the given batch size of less if the
        batch size is bigger then what we have in history

        Parameters:
        -----------
        batch_size : int
            Size of the mini-batch

        Returns:
        --------
        [e] : batch of experiences

        """
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        """
        Getter of the buffer size

        Returns:
        --------
        int : size of the hisory
        """
        return self.buffer_size

    def add(self, state, action, reward, new_state):
        """
        Add an experience to the hisotry

        Parameters:
        -----------
        state : []
            List of 2 elements containing the current state
        action: float
            action performed
        reward : float
            reward obtained
        new_state: []
            The state in which the agent arrives after doing the action from
            the state
        """
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        """
        Returns:
        --------
        If buffer is full, return buffer size.
        Otherwise, return experience counter
        """
        return self.num_experiences

    def erase(self):
        """
        Erase the history
        """
        self.buffer = deque()
        self.num_experiences = 0
