import numpy as np

from nnfw.backend import cp

class Activation:
    @staticmethod
    def relu(X):
        """
        ReLu activation function.

        Args:
            X: Input array of shape (d, N)
        Returns:
            Output array of shape (d, N)
        """
        return np.maximum(0, X)
    @staticmethod
    def relu_derivative(G, X, Y):
        """
        ReLu derivative function.

        Args:
            G: Incoming gradient of shape (d, N)
            Y: Input array of shape (d, N) holding layer output
        Returns:
            Output array of shape (d, N)
        """
        return G * (Y > 0)

    @staticmethod
    def softmax(X):
        """
        Softmax activation function.

        Args:
            X: Input array of shape (K, N)
        Returns:
            Output array of shape (K, N)
        """
        exp_Z = np.exp(X - np.max(X, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    @staticmethod
    def softmax_derivative(G, X, Y):
        """
        Args:
            G: Incoming gradient (not used for cross-entropy + softmax case)
            X: Input array (pre-activation scores)
            Y: One-hot-encoded labels
        Returns:
            P - Y (the proper gradient for cross-entropy + softmax)
        """
        P = Activation.softmax(X)
        return P - Y  # Return P - Y_labels directly

class SequenceActivation:
    @staticmethod
    def sigmoid(X):
        """
        Sigmoid activation function.

        Args:
            X: Input array of shape (d, N)
        Returns:
            Output array of shape (d, N)
        """
        return 1 / (1 + cp.exp(-X))
    @staticmethod
    def sigmoid_derivative(**kwargs):
        """
        Sigmoid derivative function.
        
        Args:
            G: Incoming gradient of shape (d, N)
            Y: Input array of shape (d, N) holding layer output
        Returns:
            Output array of shape (d, N)
        """
        G = kwargs['G']
        Y = kwargs['Y']
        return G * Y * (1 - Y)
    @staticmethod
    def softmax(X, T=1.0):
        """
        Softmax activation function, optionally with temperature scaling.

        Args:
            X (cp.ndarray): Input array of shape (K, N)
            T (float, optional): Temperature [0, 1].
        Returns:
            cp.ndarray: Output array of shape (K, N)
        """
        if T is not None:
            if T < 0 or T > 1.0:
                raise ValueError("Softmax temperature must be in the range [0, 1].")
            if T == 0:
                # Potentially add protection against 0 division
                pass
        # Catch None case
        if T is None:
            T = 1.0
        # Apply temperature scaling (note that T=1.0 is same as no temperature)
        shifted_X = (X - cp.max(X, axis=0, keepdims=True)) / T
        exp_Z = cp.exp(shifted_X)
        return exp_Z / cp.sum(exp_Z, axis=0, keepdims=True)
    @staticmethod
    def softmax_derivative(**kwargs):
        """
        Args:
            X: Input array (pre-activation scores)
            Y: One-hot-encoded labels
        Returns:
            P - Y (the proper gradient for cross-entropy + softmax)
        """
        X = kwargs['X']
        Y = kwargs['Y']
        P = SequenceActivation.softmax(X=X)
        return P - Y  # Return P - Y_labels directly
    
    @staticmethod
    def tahn(X):
        """
        Tanh activation function.

        Args:
            X: Input array of shape (d, N)
        Returns:
            Output array of shape (d, N)
        """
        return cp.tanh(X)
    @staticmethod
    def tahn_derivative(**kwargs):
        """
        Tanh derivative function.

        Source: https://medium.com/@vikrampande783/derivative-of-sigmoid-and-tanh-ffdea6efb464

        Args:
            G: Incoming gradient of shape (d, N)
            Y: Input array of shape (d, N) holding layer output
        Returns:
            Output array of shape (d, N)
        """
        G = kwargs['G']
        Y = kwargs['Y']
        return G * (1 - Y**2)
    
    @staticmethod
    def linear(X, **kwargs):
        """
        Linear activation function (identity).

        Args:
            X: Input array of shape (d, N)
        Returns:
            Output array of shape (d, N)
        """
        return X
    @staticmethod
    def linear_derivative(**kwargs):
        """
        Linear derivative function.

        Args:
            G: Incoming gradient of shape (d, N)
            Y: Input array of shape (d, N) holding layer output
        Returns:
            Output array of shape (d, N)
        """
        G = kwargs['G']
        return G
    
    @staticmethod
    def relu(X, **kwargs):
        return cp.maximum(0, X)
    @staticmethod
    def relu_derivative(**kwargs):
        G = kwargs['G']
        Y = kwargs['Y']
        return G * (Y > 0)
