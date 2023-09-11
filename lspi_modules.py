import numpy as np
from torch.nn import Sequential


class LSPI:
    def __init__(self, basis_function, discount, action_space_size=2):
        self.action_space_size = action_space_size
        self.basis_function = basis_function
        self.discount = discount
        self.delta = 0.001  # Precondition value
        # Initialize the weight vector
        self.weights = np.random.uniform(-0.1, 0.1, self.basis_function.size())
        # Initialize inverse of matrix A and vector b
        self.A_inverse = np.eye(self.basis_function.size())
        np.fill_diagonal(self.A_inverse, 1.0 / self.delta)
        self.b_vector = np.zeros((self.basis_function.size(), 1))

        # add a replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = 10000
        self.replay_buffer_index = 0
        self.batch_size = 1

    def learningStep(self, previous_state, previous_action, reward, current_state, current_action):
        basis_size = self.basis_function.size
        prev_action_one_hot = np.zeros(self.action_space_size)
        prev_action_one_hot[previous_action] = 1
        phi_previous = self.basis_function.evaluate(previous_state, prev_action_one_hot).reshape((-1, 1))

        if current_state is not None:  # Use None to represent a terminal state
            cur_action_one_hot = np.zeros(self.action_space_size)
            cur_action_one_hot[current_action] = 1
            phi_current = self.basis_function.evaluate(current_state, cur_action_one_hot).reshape((-1, 1))
        else:
            phi_current = np.zeros((basis_size, 1))

        A1_vector = np.dot(self.A_inverse, phi_previous)
        g_vector = (phi_previous - self.discount * phi_current).T

        self.A_inverse -= np.dot(A1_vector, np.dot(g_vector, self.A_inverse)) / (1 + np.dot(g_vector, A1_vector))
        self.b_vector += phi_previous * reward
        self.weights = np.dot(self.A_inverse, self.b_vector).reshape((-1,))

    def update(self, state, action, reward, next_state):
        self.learningStep(state, action, reward, next_state, self.policy(next_state))

    def policy(self, state):
        # Return the action that maximizes the Q-value
        if state is None:
            return None
        else:
            # one-hot encoding of the action that maximizes the Q-value
            # qet the Q-values of every action using the one-hot encoding of the action
            q_values = []
            for action in range(self.action_space_size):
                action_one_hot = np.zeros(self.action_space_size)
                action_one_hot[action] = 1
                phi = self.basis_function.evaluate(state, action_one_hot)
                q_values.append(np.dot(self.weights, phi))
            return np.argmax(q_values)

    def Qvalue(self, state, action):
        phi = self.basis_function.evaluate(state, action)
        return np.dot(self.weights, phi)


class LinearBasisFunction:
    def __init__(self):
        self.size = 5  # 4 dimensions for the state and 1 for the action

    def evaluate(self, state, action):
        return np.concatenate([state, [action]])


from scipy.spatial.distance import cdist

class RadialBasisFunction:
    def __init__(self, centers, gamma):
        self.centers = centers
        self.gamma = gamma

    def evaluate(self, state, action):
        features = np.concatenate([state.raw_state, [action.raw_action]])
        diff = cdist(features[None, :], self.centers, 'euclidean')
        return np.exp(-self.gamma * diff**2)


class BlockBasis(object):
    """Basis functions over (s,a) which activates the appropriate block of state feature for each action.
    All other blocks have 0 values.

    Parameters
    ----------
    """

    def __init__(self, num_actions, state_dim):
        """Initialize BlockBasis.
        """

        self.num_actions = num_actions
        self.state_dim = state_dim

    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        # return self.state_dim * self.num_actions
        return self.state_dim * self.num_actions + 1  # Add 1 for a single bias

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a single active block of state features.
        """
        phi = np.zeros(self.size())
        phi[action * self.state_dim:(action + 1) * self.state_dim] = state
        phi[-1] = 1  # Bias
        return phi


class PolynomialBasisFunction:
    def __init__(self, degree):
        self.degree = degree

    def size(self):
        return (self.degree + 1) ** 2

    def evaluate(self, state, action):
        # concatenate the state and action
        features = np.concatenate([state, [action]])
        # compute the polynomial features
        phi = np.array([features[0] ** i * features[1] ** j for i in range(self.degree + 1) for j in range(self.degree + 1)])
        return phi

class IdentityBasisFunction:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def size(self):
        return self.state_size + self.action_size

    def evaluate(self, state, action):
        concat = np.concatenate([state, action])
        return concat



