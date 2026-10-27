import numpy as np

from nnfw.layers.base import Layer

class FlattenLayer(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        """
        Flatten in foward pass.

        Args:
            X: Input data
        """
        self.input_shape = X.shape
        #print(f"student: shape before flatten: {X.shape}")
        X = np.transpose(X, (2, 0, 1, 3))
        X = X.reshape(-1, X.shape[-1], order='C')
        #print(f"student: shape after flatten: {X.shape}")
        return X
    def backward(self, G):
        """
        Unflatten in backward pass.

        Args:
            G: Incoming gradient of shape (d, N)
        """        
        # Reshape back to (C, H, W, N)
        G = G.reshape((self.input_shape[2], self.input_shape[0], self.input_shape[1], self.input_shape[3]), order='C')
        # Transpose back to (H, W, C, N)
        return np.transpose(G, (1, 2, 0, 3))
