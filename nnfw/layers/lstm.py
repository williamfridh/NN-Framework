from nnfw.activations import SequenceActivation as Activation
from nnfw.backend import cp
from nnfw.layers.base import Layer

class LSTMLayer(Layer):
    def __init__(self, net, K, m, lam = 0.):
        """
        Initializes the Long-Short-Term-Memory layer.

        Note that this implementation makes use of the concatenated matrix approach.
        This means that instead of performing four separeate matrix multiplication,
        the weights are stacked into a larger matrix.

        Args:
            net: The parent network
            K: Input dim. (num. of features)
            m: Num. of neurons in the hidden layer
            activation: The activation function to use
            lam: Regularization strength
        """
        super().__init__(net, lam)

        self.K = K
        self.m = m

        self.cache = []

        # Weights are stacked in order f, i, c, o, shape = (4m, m + K)
        self.W = (1/cp.sqrt(2*m))*self.net.rng.standard_normal(size = (4*m, m + K))
        self.b = cp.zeros((4*m, 1))
        # Set the Forget Gate biases (the first "m" elements) to 1.0 for faster training
        self.b[:m, :] = 1.0
        
        self.grad_W = cp.zeros((4*m, m + K))
        self.grad_b = cp.zeros((4*m, 1))

    def clear(self, force = False):
        # Save the very last hidden state
        h_last = self.Z[-1] if len(self.Z) > 0 else None
        c_last = self.C[-1] if hasattr(self, "C") and len(self.C) > 0 else None
        
        # Clear the histories
        self.X = []
        self.Y = []
        self.cache = []
        
        # Reset Z to only contain the last state
        if h_last is not None and not force:
            self.Z = [h_last]
            self.C = [c_last]
        else:
            self.Z = []
            self.C = []

    def forward(self, X = None, h_prev=None, c_prev=None, save_state=True):
        """
        Forward pass.

        Args:
            X: Input data (sequence) of shape (K, N)

        Returns:
            Output of shape (K, N)
            Hidden state of shape (m, N)
        """

        if X is None and save_state and self.prev_layer is not None:
            X = self.prev_layer.get_Z()

        m = self.W.shape[0] // 4

        # Get prev. hidden state (first time step)
        if h_prev is None:
            h_prev = self.Z[-1] if len(self.Z) > 0 else cp.zeros((m, X.shape[1]))
            c_prev = self.C[-1] if len(self.C) > 0 else cp.zeros((m, X.shape[1]))
        else:
            # If h_prev is provided (sythesis), we assume c_prev is 0 for simplicity
            if c_prev is None:
                c_prev = cp.zeros((m, X.shape[1]))

        # Concatenate prev. hidden state and current x (input)
        z_t = cp.concatenate((h_prev, X), axis=0)  # Shape (m + K, 1)

        A_t = self.W @ z_t + self.b

        # Split A_t into the four gates: f, i, c, and o
        a_f, a_i, a_c, a_o = cp.split(A_t, 4, axis=0)

        f_t = Activation.sigmoid(a_f)
        i_t = Activation.sigmoid(a_i)
        c_tilde = Activation.tahn(a_c) # Note that cell state uses tahn activation
        o_t = Activation.sigmoid(a_o)

        # Upodate cell and hidden states
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * Activation.tahn(c_t)

        if save_state:
            self.X.append(X)
            self.Y.append(A_t)
            self.Z.append(h_t)
            self.C.append(c_t)
            # Save everything needed for backprop
            self.cache.append({
                'z': z_t,
                'f': f_t,
                'i': i_t,
                'c_tilde': c_tilde,
                'o': o_t,
                'c_prev': c_prev,
                'c': c_t,
                'h': h_t
            })
            return h_t
        
        return h_t, c_t

    def backward(self):
        # Get time steps and prepare grad. arrays
        T = len(self.X)

        self.grad_W = cp.zeros_like(self.W)
        self.grad_b = cp.zeros_like(self.b)

        m = self.W.shape[0] // 4
        batch_size = self.X[0].shape[1]
        
        dh_next = cp.zeros((m, batch_size))
        dc_next = cp.zeros((m, batch_size))

        G_prev = self.next_layer.G
        G_to_pass_down = [None] * T

        for t in reversed(range(T)):
            vars = self.cache[t]

            # Sum gradient from prev. layer and total
            dh_t = G_prev[t] + dh_next
            
            tanh_c = Activation.tahn(vars['c'])
            dc_t = dc_next + Activation.tahn_derivative(G=dh_t * vars['o'], Y=tanh_c)

            # Calculate pre-activation gradients for each gate
            da_f = Activation.sigmoid_derivative(G=dc_t * vars['c_prev'], Y=vars['f'])
            da_i = Activation.sigmoid_derivative(G=dc_t * vars['c_tilde'], Y=vars['i'])
            da_c = Activation.tahn_derivative(G=dc_t * vars['i'], Y=vars['c_tilde'])
            da_o = Activation.sigmoid_derivative(G=dh_t * tanh_c, Y=vars['o'])

            # Stack and compute the weights gradients
            dA_t = cp.concatenate((da_f, da_i, da_c, da_o), axis=0)

            self.grad_W += dA_t @ vars['z'].T
            self.grad_b += cp.sum(dA_t, axis=1, keepdims=True)

            # And pass gradients back to earlier layers/timesteps
            dz_t = self.W.T @ dA_t

            m = self.W.shape[0] // 4
            dh_next = dz_t[:m, :]
            dx_t = dz_t[m:, :]

            dc_next = dc_t * vars['f']
            G_to_pass_down[t] = dx_t

        # Average gradients over time steps * batch size
        self.grad_W /= (T * self.X[0].shape[1])
        self.grad_b /= (T * self.X[0].shape[1])

        # Prevent exploding gradients torugh clipping
        self.grad_W = cp.clip(self.grad_W, -1.0, 1.0)
        self.grad_b = cp.clip(self.grad_b, -1.0, 1.0)

        # Save gradients to pass to layer below (if any)
        self.G = G_to_pass_down
    
    def update_params_adam(self, eta, t):
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
