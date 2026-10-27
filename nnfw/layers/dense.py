import numpy as np

from nnfw.activations import Activation
from nnfw.layers.base import Layer

rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state

class FullyConnectedLayer(Layer):
    def __init__(self, d, m, activation, lam = 0.):
        """
        Initializes the fully connected layer using He initilization
        for both weights and biases.

        Args:
            d: Input dim. (num. of features)
            m: Num. of neurons in the hidden layer
            activation: The activation function to use
            lam: Regularization strength
        """
        
        self.W = (1/np.sqrt(d))*rng.standard_normal(size = (m, d)) # He initialization
        self.b = np.zeros((m, 1))
        self.lam = lam

        if activation == "relu":
            self.activation_func = Activation.relu
            self.activation_func_derivative = Activation.relu_derivative
        elif activation == "softmax":
            self.activation_func = Activation.softmax
            self.activation_func_derivative = Activation.softmax_derivative
        else:
            print(f"Unsupported activation detected. Defaulting to ReLU.")
            self.activation_func = Activation.relu
            self.activation_func_derivative = Activation.relu_derivative


        self.Y = None
        self.N = None
        self.X = None

        self.grad_W = np.array([])
        self.grad_b = np.array([])

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input data of shape (d, N)

        Returns:
            Output of shape (m, N)
        """
        self.Y = self.W @ X + self.b
        self.Z = self.activation_func(self.Y)
        self.N = X.shape[1]
        self.X = X
        return self.Z
    
    def backward(self, G, Y = None):
        """
        Backward pass.

        Args:
            G: Incoming gradient of shape (m, N)
            eta: Learning rate

        Returns:
            G_pass: Gradient to pass to the previous layer, of shape (d, N)
            W: Updated weights
            b: Updated biases
        """

        # Pass correct parameters to the activation function derivative
        if self.activation_func.__name__ == 'softmax' and Y is not None:
            G = self.activation_func_derivative(G, self.Y, Y)
        else:
            G = self.activation_func_derivative(G, self.Y, self.Z)

        G_pass = self.W.T @ G

        self.grad_W = (1 / self.N) * G @ self.X.T + 2 * self.lam * self.W
        self.grad_b = (1 / self.N) * np.sum(G, axis=1, keepdims=True)

        return G_pass, self.W, self.b
    
    def update_params(self, eta):
        """
        Updates the weights and biases using the computed gradients.

        Args:
            eta: Learning rate
        """
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b
