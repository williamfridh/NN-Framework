from nnfw.backend import cp
from nnfw.layers.base import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, net, d, m, activation, lam = 0.):
        """
        Initializes the fully connected layer using He initilization
        for both weights and biases.

        W: (neurons in net., input. dim.) = (m, d)
        b: (neurons in net., 1) = (m, 1)

        Args:
            net: The neural network to which this layer belongs
            d: Input dim. (num. of features)
            m: Num. of neurons in the hidden layer
            activation: The activation function to use
            lam: Regularization strength
        """
        super().__init__(net, lam, activation)

        self.d = d
        self.m = m

        self.W = (1/cp.sqrt(m))*self.net.rng.standard_normal(size = (m, d)) # He initialization
        self.b = cp.zeros((m, 1))

        self.grad_W = \
        self.grad_b = None

    def clear(self, **kwargs):
        self.X = []
        self.Y = []
        self.Z = []

    def forward(self, X = None, save_state=True, T=None, theta=None):
        """
        Forward pass.   

        Args:
            X: Input data of shape (d, N)
            save_state (boolean): Determined if it should save state to layer
            T (float): Temperature in range [0, 1]
        """
        if X is None and save_state and self.prev_layer is not None:
            X = self.prev_layer.get_Z()

        # Store X, Y, Z or return Z
        #print(f"self.W.shape: {self.W.shape}, X.shape: {X.shape}, self.b.shape: {self.b.shape}")
        Y = self.W @ X + self.b
        Z = self.activation_func(X=Y, T=T)

        if theta is not None:
            """
            Note that this section was first written using a loop but was
            changed drastically after code review performed by Gemini.
            """
            # 1. Sort in descending order (note - sign)
            Z_idx_desc = cp.argsort(-Z, axis=0)
            # 2. Reorder Z using decending indicies
            Z_sorted = cp.take_along_axis(Z, Z_idx_desc, axis=0)
            # 3. Calculate cumulative sum down the columns
            cum_sum = cp.cumsum(Z_sorted, axis=0)
            # 4. Create a mask for removing elements
            # The goal is remove all elements efter the crossing of the
            # threshold and shifting it down by one to keep the element 
            # crossing it
            mask = cum_sum >= theta
            mask = cp.roll(mask, shift=1, axis=0)
            mask[0, :] = False # Prevent removal of top probaiblity even if it's > theta
            # 5. Apply the mask
            Z_sorted[mask] = 0.
            # 6. Restore the propabilities into original order
            Z = cp.zeros_like(Z)
            cp.put_along_axis(Z, Z_idx_desc, Z_sorted, axis=0)
            # 7. Renormalize
            Z = Z / cp.sum(Z, axis=0, keepdims=True)

        if save_state:
            self.X.append(X)
            self.Y.append(Y)
            self.Z.append(Z)
        return Z
    
    def backward(self, Y):
        # Get time steps and prepare grad. arrays
        T = len(self.X)
        self.grad_W = cp.zeros_like(self.W)
        self.grad_b = cp.zeros_like(self.b)
        
        # And a list of grads (dL/dht) for each timestep to pass
        G = [None] * T # List to store dL/dht for each timestep

        # Go trough time
        for t in reversed(range(T)):
            
            # Calc. gradient
            g_t = self.activation_func_derivative(G = self.Y[t])
            
            # Adjust weight and bias gradients
            self.grad_W += g_t @ self.X[t].T
            self.grad_b += cp.sum(g_t, axis=1, keepdims=True)
            
            # Compute dL/dh_t to pass back to next layer
            G[t] = self.W.T @ g_t

        # Average gradients over time steps * batch size and store grads.
        self.grad_W /= (T * self.X[0].shape[1])
        self.grad_b /= (T * self.X[0].shape[1])
        self.G = G
    
    def update_params(self, eta):
        """
        Updates the weights and biases using the computed gradients.

        Args:
            eta: Learning rate
        """
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b

    def update_params_adam(self, eta, t):
        """
        Updates the weights and biases using the computed gradients.

        Args:
            eta: Learning rate
        """
        beta1 = 0.9
        beta2 = 0.999

        if not hasattr(self, "W_m"):
            self.W_m = cp.zeros_like(self.W)
            self.W_v = cp.zeros_like(self.W)
        
            self.b_m = cp.zeros_like(self.b)
            self.b_v = cp.zeros_like(self.b)

        # Weight updates
        # Update bias moment estimate
        self.W_m = beta1 * self.W_m + (1 - beta1) * self.grad_W
        self.W_v = beta2 * self.W_v + (1 - beta2) * (self.grad_W**2)
        # Compute the bias corrected estimate
        w_m_hat = self.W_m / (1 - beta1**t)
        w_v_hat = self.W_v / (1 - beta2**t)
        # Update params
        self.W -= eta * w_m_hat / (cp.sqrt(w_v_hat) + 1e-8)

        # Bias updates
        # Update bias moment estimate
        self.b_m = beta1 * self.b_m + (1 - beta1) * self.grad_b
        self.b_v = beta2 * self.b_v + (1 - beta2) * (self.grad_b**2)
        # Compute the bias corrected estimate
        b_m_hat = self.b_m / (1 - beta1**t)
        b_v_hat = self.b_v / (1 - beta2**t)
        # Update params
        self.b -= eta * b_m_hat / (cp.sqrt(b_v_hat) + 1e-8)
