""" A committee for NN prediction """
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing.data import StandardScaler as StandardScalar

def scale_func(x, xmax):
    return 0.5 + 0.5*np.tanh(2.0*(x-xmax))


def quartic_potential(x, E0=1.0, C=1.0/120.0):
    """
    model quartic potential in kcal/mol
    """
    x2 = x**2.0
    return E0*(C*x2**2.0 - x2), 0.0

class ModelNetwork(object):
    """
    Learn a 1-D model function with scikit-learn
    """
    def __init__(self, depth=4):
        self._nn = MLPRegressor(hidden_layer_sizes=(11,)*depth)

    def train(self, X, y):
        """
        train to a data set of x and y values
        :type X: list of lists of X values (each example should be of length 1)
        :type y: list of y values
        """

        assert len(X) == len(y)
        assert all([len(example) == 1 for example in X])

        self.scaler = StandardScalar()
        self.scaler.fit(X)
        self.ymean = np.mean(y)
        self.yvar = np.std(y)
        self._nn.fit(self.scaler.transform(X), (np.array(y) - self.ymean) / self.yvar)

    def predict(self, X):
        """
        predict a y value at a single X value
        :type X: float 
        """
        y = self._nn.predict(self.scaler.transform([[X]]))[0]
        return y*self.yvar + self.ymean

class ModelCommittee(object):
    """
    A committee for a 1-D model function
    """

    def __init__(self, nmembers=10, smax=5.0):
        """
        Setup a committee 
        :type nmembers: int, number of members in the committee
        :type smax: float, hard cap on sigma
        """

        self._committee = [ModelNetwork() for i in range(nmembers)]
        self._smax = smax

    def train(self, X, y):
        """
        train the commitee
        :type X: list of lists of X values (each example should be of length 1)
        :type y: list of y values
        """
        for member in self._committee:
            member.train(X, y)
    

    def predict(self, X):
        """
        Predict a mean and std dev from the committee
        """
        predictions = np.array([member.predict(X) for member in self._committee])

        # put a smooth cap on sigma
        s = np.std(predictions)
        switch = scale_func(s, self._smax)
        f = (1.0 - switch)*s + switch*self._smax

        return np.mean(predictions), f 
        
        
        
