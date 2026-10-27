import numpy as np

from nnfw.activations import Activation
from nnfw.layers.base import Layer

rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state

class ConvolutionalLayer(Layer):
    def __init__(self, f, n_f, activation, lam = 0.):
        """
        Initializes the convolutional layer.

        Args:
            f: Filter size (4 for 4x4 filters)
            n_f: Num. of filters
            activation: The activation function to use
            lam: Regularization strength
        """
        self.f = f
        self.n_f = n_f
        self.lam = lam
        self.n = None
        self.n_p = (32 // f) ** 2 # Num. of patches per image

        self.b = np.zeros((n_f, 1))
        self.b_grads = np.zeros_like(self.b)

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

        self.Fs = (1/np.sqrt(f*f*3))*rng.standard_normal(size=(f, f, 3, n_f)) # He initialization
        self.Fs_grads = np.zeros_like(self.Fs)

    def compute_mx(self, X, f, n):
        """
        Computes the MX matrix for convolution using matrix multiplication.
        This function was created earlier during testing phase.

        Args:
            X: Input data of shape (32, 32, 3, n)
            f: Filter size
            n: Number of images

        Returns:
            MX: Matrix of shape (n_p, f**2 * 3, n)
        """
        MX = np.zeros((self.n_p, f**2 * 3, n))
        # For each image
        for i in range(n):
            # We iterate though each non-overlapping patch
            l = 0 #lth sub-region
            for x in range(0, 32, f):
                for y in range(0, 32, f):
                    # We extract the patch
                    X_patch = X[x:x+f, y:y+f, :, i]
                    # And store is using given instructions
                    MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')
                    l += 1
        self.MX = MX

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input data of shape (d, N)
        """
        self.n = X.shape[1]
        x_img = X.reshape((32, 32, 3, self.n), order='F')
        self.compute_mx(x_img, self.f, self.n)
        Fs_flat = self.Fs.reshape((self.f*self.f*3, self.n_f), order='C')
        out = np.einsum('pkn,kf->pfn', self.MX, Fs_flat, optimize=True)
        self.Y = out.reshape((32//self.f, 32//self.f, self.n_f, self.n), order='C')
        self.Y += self.b  # Add bias
        self.Z = self.activation_func(self.Y)
        return self.Z
    
    def backward(self, G):
        """
        Compute the gradients with respect to each filter.mro
        
        Args:
            n: Number of images
            MX: The matrix computed in the forward pass

        """

        # First apply derivative of activation function
        G = self.activation_func_derivative(G, self.Y, self.Z)

        # Then we compute the filter gradients
        GG = G.reshape((self.n_p, self.n_f, self.n), order='C')

        MXt = np.transpose(self.MX, (1, 0, 2)) # (provided)
        grad_Fs_flat = np.einsum('ijn, jln ->il', MXt, GG, optimize=True) # (provided)
        
        self.Fs_grads = (1 / self.n) * grad_Fs_flat.reshape(self.Fs.shape, order='C')
        self.b_grads = (1 / self.n) * np.sum(GG, axis=(0, 2), keepdims=True).reshape(self.b.shape) + 2 * self.lam * self.b

        #print(f"Fs_grads.shape: {self.Fs_grads.shape}")

        # Finally we compuate and return input gradient
        Fs_flat = self.Fs.reshape((self.f*self.f*3, self.n_f), order='C')
        G_X_flat = np.einsum('kf, pfn ->pkn', Fs_flat, GG, optimize=True)
        G_X = G_X_flat.reshape((-1, self.n), order='C')
        return G_X
    
    def update_params(self, eta):
        self.Fs -= eta * self.Fs_grads
        self.b -= eta * self.b_grads
