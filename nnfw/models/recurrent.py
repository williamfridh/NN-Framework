import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import ipywidgets as widgets
from IPython.display import display

from nnfw.backend import cp
from nnfw.layers.lstm import LSTMLayer
from nnfw.layers.recurrent import RecurrentLayer
from nnfw.layers.sequence_dense import FullyConnectedLayer

def print_free_mem():
    """
    Print free memoery (Gigabytes).

    Important function for debugging. Make sure to print frequently when
    performing multiple epochs or re-runs.

    Returns:
        Amount of free memoery (float)
    """
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024 ** 3)
    print(f"Available RAM: {free_gb:.2f} GB")
    return free_gb

class RecurrentNeuralNetwork:
    """
    Recurrent Neural Network class. Collect all relevant methods here for simplicity and modularity.

    Attributes:
        d: Number of features (input dimension)
        m: Number of hidden units
        net: Dictionary containing the weights and biases of the network
        lam: Regularization strength
    """

    def __init__(self, train_data_handler=None, val_data_handler=None, test_data_handler=None):
        """
        Initialize network with weights and biases.

        Args:
            d: Number of features (input dimension)
            m: Number of hidden units
            lam: Regularization strength
        """
        self.layers = []
        self.input_layer =  self.output_layer = None

        self.h = None

        self.avg_loss = 0.

        self.training_loss_history = \
            self.validation_loss_history = \
            self.training_cost_history = \
            self.validation_cost_history = \
            self.training_acc_history = \
            self.validation_acc_history = None

        # We allways keep a new random number generator in the class for reproducibility
        self.rng = cp.random.default_rng()
        # get the BitGenerator used by default_rng
        BitGen = type(self.rng.bit_generator)
        # use the state from a fresh bit generator
        seed = 42
        self.rng.bit_generator.state = BitGen(seed).state

        self.smooth_loss_history = []

        self.train_data_handler  = train_data_handler
        self.val_data_handler    = val_data_handler
        self.test_data_handler   = test_data_handler

        if self.train_data_handler is not None:
            self.train_data_handler.rndm = self.rng
        if self.val_data_handler is not None:
            self.val_data_handler.rndm = self.rng
        if self.test_data_handler is not None:
            self.test_data_handler.rndm = self.rng

    def get_params(self):
        """
        Get weights and biases for each layer.
        This function is used for compatability with
        the teachers provided testing code.

        Note:
        We must fetch the values using get() to move the data from
        GPU to CPU.
        """
        net = [] # List holding dict(W=None, b=None, U=None, V=None, c=None)
        for layer in self.layers:
            l = {'activation': layer.activation}
            if layer.__class__.__name__ == "FullyConnectedLayer":
                l['type'] = 'FullyConnected'
                l['W'] = layer.W.get() if hasattr(layer.W, "get") else layer.W
                l['b'] = layer.b.get() if hasattr(layer.b, "get") else layer.b
                l['d'] = layer.d
                l['m'] = layer.m
            if layer.__class__.__name__ == "RecurrentLayer":
                l['type'] = 'Recurrent'
                l['W'] = layer.W.get() if hasattr(layer.W, "get") else layer.W
                l['U'] = layer.U.get() if hasattr(layer.U, "get") else layer.U
                l['b'] = layer.b.get() if hasattr(layer.b, "get") else layer.b
                l['K'] = layer.K
                l['m'] = layer.m
            if layer.__class__.__name__ == "LSTMLayer":
                l['type'] = 'LSTM'
                l['W'] = layer.W.get() if hasattr(layer.W, "get") else layer.W
                l['b'] = layer.b.get() if hasattr(layer.b, "get") else layer.b
                l['K'] = layer.K
                l['m'] = layer.m
            net.append(l)
        return net
    
    def save_model(self, file_name):
        """
        Save model parameters to a file.

        Args:
            file_name: The name of the file to save the model parameters to
        """
        params = self.get_params()
        np.save(file_name, params)
        print(f"Model parameters saved to {file_name}.")
    
    def predict(self, X):
        """
        Predict.

        Args:
            X: data (dim., batch size, seq. len.)
        Return:
            P_final: Predicted probabilities of shape (K, N)
        """
        for l in self.layers:
            l.clear(force = True)

        X = cp.asarray(X)
        
        seq_len = X.shape[2] # seq_len = N
        
        for i in range(seq_len):
            current_X = X[:, :, i]
            for layer_i, layer in enumerate(self.layers):
                layer.forward(current_X if layer_i == 0 else None)

        P_final = self.output_layer.get_Z()
        return P_final

    def add_layer(self, layers):
        """
        Add a layer to the network.

        Args:
            layer: An instance of a layer class
        """
        for l in layers:
            # Make connections before appending to the layers list
            l.prev_layer = self.layers[-1] if self.layers else None
            if len(self.layers) > 0:
                self.layers[-1].next_layer = l
            else:
                self.input_layer = l
            self.output_layer = l
            # Append to layer list
            self.layers.append(l)

    def reset_layers(self):
        """
        Reset all layers.

        Empties X, Y and Z of each layer.
        """
        for l in self.layers:
            l.X = l.Y = l.Z = []

    def compute_mse(self, P, Y):
        """
        Compute mean squared error.

        Args:
            P: Array of shape (K, N) containing the predicted values for each class and sample
            Y: Array of shape (K, N) containing the true values for each class and sample
        Returns:
            loss: The computed mean squared error value
        """
        P = cp.asarray(P)
        Y = cp.asarray(Y)
        # Compute mean squared error
        loss = cp.mean((P - Y) ** 2)
        return float(loss.get() if hasattr(loss, "get") else loss)
    
    def compute_acc(self, P, y):
        """
        Compute accuracy of predictions.

        Args:
            P: Array of shape (K, N) containing the predicted probabilities for each class and sample
            y: Array of shape (N,) containing the integer labels for each sample
    
        Returns:
            acc: The computed accuracy value
        """
        predicted_labels = cp.argmax(P, axis=0)
        acc = cp.mean(predicted_labels == y)
        return float(acc.get() if hasattr(acc, "get") else acc)
    
    def test(self):
        """
        Test Accuracy.

        Return:
            avg_loss: Average loss over sequence and batches
        """
        total_loss = 0.0
        steps = 0
        generator = self.test_data_handler.get_generator()

        for X_batch, Y_batch in generator:
            X_batch_gpu = cp.asarray(X_batch)
            Y_batch_gpu = cp.asarray(Y_batch)

            total_loss += self.forward_pass(
                X_batch_gpu,
                Y_batch_gpu,
                force_clean=True,
                store_loss=False
            )
            steps += 1

        avg_loss = total_loss / steps
        print(f"Avg. loss: {avg_loss:.4f}")
        return avg_loss
    
    def forward_pass(self, X, Y, force_clean=False, store_loss=True):
        """
        Network Forward Pass.

        Args:
            X: data (dim., batch size, seq. len.)
            Y: labels (dim., batch size, seq. len.)
            force_clean: Force layer cleaning
            store_loss: Determined if loss should be stored in self.avg_loss

        Return: 
            avg_loss: Average loss over sequence and batches
        """

        for l in self.layers:
            l.clear(force = force_clean)
        
        seq_len = X.shape[2] # seq_len = N
        
        for i in range(seq_len):
            current_X = X[:, :, i]
            for layer_i, layer in enumerate(self.layers):
                layer.forward(current_X if layer_i == 0 else None)

        # Grap last prediction
        P_final = self.output_layer.get_Z()

        avg_loss = self.compute_mse(P_final, Y)
        if store_loss:
            self.avg_loss = avg_loss
        return avg_loss
        
    def backward_pass(self, Y):
        """
        Network Backward Pass.

        Args:
            Y (array): labels (dim., batch size, seq. len.)
        """

        T           = self.train_data_handler.seq_length
        batch_size  = self.train_data_handler.batch_size

        # Calculate MSE grad. for output layer (final pred.)
        P_final = self.output_layer.get_Z()
        dL_dP = 2 * (P_final - Y) / (T * batch_size)

        # We build a list of gradients for BPTT and zero all but the last
        Y_grads = [cp.zeros_like(dL_dP) for _ in range(T)]
        Y_grads[-1] = dL_dP

        # Iterate backward through layers (FullyConnected then Recurrent)
        for layer in reversed(self.layers):
            if isinstance(layer, FullyConnectedLayer):
                # Calculate gradients at each timestep
                layer.backward(Y_grads)
            elif isinstance(layer, RecurrentLayer):
                # Performs BPTT using the list of gradients
                layer.backward()
            elif isinstance(layer, LSTMLayer):
                layer.backward()

    def train(self, learning_rate, epochs, best_model_file_name = None):
        """
        Train.

        Trains the RNN using shuffled chunks of data holding sequences.

        Args:
            learning_rate (int): learning rate of training
            epochs (int): number of epochs
            best_model_file_name (str): file name for storing best found model (state with lowest avg. loss)
        """

        seq_len = self.train_data_handler.seq_length #sequence length, longer means smaller chunks

        # Record model
        best_loss = float('inf')
        best_params = None

        update_step = 0
        for epoch_i in range(epochs):
            e = 0
            self.reset_layers()
            print_free_mem()

            # Start timer and display progress bar
            start_time = time.perf_counter()
            bar = widgets.IntProgress(
                value=0,
                min=0,
                max=self.train_data_handler.get_steps_per_epoch(),
                description=f"Epoch {epoch_i + 1}/{epochs}:",
                bar_style='info',
                orientation='horizontal'
            )
            display(bar)

            # Reset layers before epoch
            self.reset_layers()

            """
            X starts as (m, K) = (dim, seq. len.)
            parallel_X then becomes a tuple of B arrays shaped like X
            _X_batch then stacks among axis 1, creating shape (m, B, K) = (dim., batch size, seq. len.)
            """
            for X_batch, Y_batch in self.train_data_handler.get_generator():
                """
                Because the data lives on the CPU, we must move it to the GPU's VRAM.
                """
                X_batch_gpu = cp.asarray(X_batch)
                Y_batch_gpu = cp.asarray(Y_batch)

                # Perform forward & backward pass
                self.forward_pass(X_batch_gpu, Y_batch_gpu)
                self.backward_pass(Y_batch_gpu)

                # Use avg_loss if update step is first,
                # otherwise calculate it
                # Note we only use smooth loss for plotting
                if update_step == 0:
                    smooth_loss = self.avg_loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * self.avg_loss

                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    best_params = self.get_params()
                # Save loss hisory and print status
                self.smooth_loss_history.append(smooth_loss)
                #print(f"Update step: {update_step} || Smooth loss: {smooth_loss:.4f}")
                # Update counters
                update_step += 1
                e += seq_len
                bar.value += 1

                # Update layer parameters
                for l in self.layers:
                    l.update_params_adam(learning_rate, update_step)

            # Validation
            print(f"\nEvaluating Validation Set for Epoch {epoch_i + 1}...")

            val_loss_total, val_steps = 0, 0

            for X_val_batch, Y_val_batch in self.val_data_handler.get_generator():
                # Move validation batch to GPU
                X_val_batch_gpu = cp.asarray(X_val_batch)
                Y_val_batch_gpu = cp.asarray(Y_val_batch)
                val_loss_total += self.forward_pass(X_val_batch_gpu, Y_val_batch_gpu, force_clean=True, store_loss=False)
                val_steps += 1

            epoch_val_loss = val_loss_total / val_steps
            print(f"Epoch {epoch_i + 1} Validation Loss: {epoch_val_loss:.4f}")

            # Save model if new best is found
            if epoch_val_loss < best_loss:
                print(f"--> Validation loss improved from {best_loss:.4f} to {epoch_val_loss:.4f}. Saving model!")
                best_loss = epoch_val_loss
                best_params = self.get_params()
                if best_model_file_name is not None:
                    self.save_model(best_model_file_name)

            # End timer and print epoch results
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Time: {execution_time:.2f}s || Update steps: {update_step} || Smooth loss: {self.smooth_loss_history[-1]:.4f}")

        # Load best model parameters
        if best_params is not None:
            self.load_parameters(params=best_params)

        # End training with printing loss history
        print(f"Best model loss: {best_loss:.4f}")
        self.test()
        self.print_loss_history()
        return best_loss

    def print_loss_history(self):
        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(self.smooth_loss_history, label='Training', color='green')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Smooth loss')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def load_parameters(self, file_name=None, params=None):
        import os

        # Clear current layers
        self.layers = []
        self.input_layer = self.output_layer = None

        if params is not None:
            weights_dict = params
        else:
            if file_name is None:
                raise ValueError("file_name must be provided when params is None")

            found = None
            try_path = os.path.expanduser(file_name)
            if os.path.exists(try_path):
                found = try_path

            if found is None:
                raise FileNotFoundError(f"Model file not found: {file_name}.")

            weights_dict = np.load(found, allow_pickle=True)

        for l in weights_dict:
            if l['type'] == 'FullyConnected':
                layer = FullyConnectedLayer(
                    net=self,
                    d=l['d'],
                    m=l['m'],
                    activation=l['activation']
                )
                layer.W = cp.asarray(l['W'])
                layer.b = cp.asarray(l['b'])
            elif l['type'] == 'Recurrent':
                layer = RecurrentLayer(
                    net=self,
                    K=l['K'],
                    m=l['m'],
                    activation=l['activation']
                )
                layer.W = cp.asarray(l['W'])
                layer.U = cp.asarray(l['U'])
                layer.b = cp.asarray(l['b'])
            elif l['type'] == 'LSTM':
                layer = LSTMLayer(
                    net=self,
                    K=l['K'],
                    m=l['m']
                )
                layer.W = cp.asarray(l['W'])
                layer.b = cp.asarray(l['b'])
            else:
                raise ValueError(f"Unknown layer type: {l['type']}")
            self.add_layer([layer])
        print("Successfully loaded and set model parameters")