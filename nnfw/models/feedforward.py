import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    """
    Neural Network class. Collect all relevant methods here for simplicity and modularity.

    Attributes:
        d: Number of features (input dimension)
        m: Number of hidden units
        net: Dictionary containing the weights and biases of the network
        lam: Regularization strength
    """

    def __init__(self):
        """
        Initialize network with weights and biases.

        Args:
            d: Number of features (input dimension)
            m: Number of hidden units
            lam: Regularization strength
        """

        #self.d = d
        #self.m = m
        #self.lam = lam
        self.layers = []
        self.output_layer = None

        self.training_loss_history = None
        self.validation_loss_history = None
        self.training_cost_history = None
        self.validation_cost_history = None
        self.training_acc_history = None
        self.validation_acc_history = None

    def print_structure(self):
        """
        Print layer structure of the network.
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            print(f"Layer #{i+1}: {layer.W[0].shape[0]} neurons, activation: {layer.activation_func.__name__}")

    def get_grads(self):
        """
        Get gradients for weights and biases for each layer.
        This function is used for compatability with provided
        teached code and testing.
        """
        grads = dict(W=[], b=[], Fs=[], Fs_b=[])
        for layer in self.layers:
            if layer.__class__.__name__ == "ConvolutionalLayer":
                grads['Fs'].append(layer.Fs_grads)
                grads['Fs_b'].append(layer.b_grads)
            elif layer.__class__.__name__ == "FullyConnectedLayer":
                grads['W'].append(layer.grad_W)
                grads['b'].append(layer.grad_b)
        return grads
    
    def get_weights_and_biases(self):
        """
        Get weights and biases for each layer.
        This function is used for compatability with
        the teachers provided testing code.
        """
        net = dict(W=[], b=[], Fs=[], Fs_b=[])
        for layer in self.layers:
            if layer.__class__.__name__ == "FullyConnectedLayer":
                net['W'].append(layer.W)
                net['b'].append(layer.b)
            elif layer.__class__.__name__ == "ConvolutionalLayer":
                net['Fs'].append(layer.Fs)
                net['Fs_b'].append(layer.b)
        return net

    def add_layer(self, layers):
        """
        Add a layer to the network.

        Args:
            layer: An instance of a layer class
        """
        for l in layers:
            self.layers.append(l)
            self.output_layer = l
        
    def apply_network(self, X):
        """
        Apply network forward pass. Stores and returns the output of each layer (after activation).

        Args:
            X: Array of shape (d, N) containing the input data

        Returns:
             layer_output: List of arrays holding the output of each layer (after activation)
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def compute_loss(self, P, y):
        """
        Compute loss using cross-entropy and L2 regularization.

        Args:
            P: Array of shape (K, N) containing the predicted probabilities for each class and sample
            y: Array of shape (N,) containing the integer labels for each sample

        Returns:
            loss: The computed loss value
        """
        loss = (1 / P.shape[1]) * -np.sum(np.log(P[y, np.arange(P.shape[1])]))
        return loss
    
    def compute_cost(self, P, y):
        """
        Compute cost using cross-entropy and L2 regularization.

        Note that this only supports weights from fully connected (dense)
        layers and not for convolutional layers.

        Args:
            P: Array of shape (K, N) containing the predicted probabilities for each class and sample
            y: Array of shape (N,) containing the integer labels for each sample

        Returns:
            cost: The computed cost value
        """
        weight_sum = 0
        for layer in self.layers:
            if layer.__class__.__name__ == "FullyConnectedLayer":
                weight_sum += np.sum(layer.W ** 2)
            elif layer.__class__.__name__ == "ConvolutionalLayer":
                weight_sum += np.sum(layer.Fs ** 2)
        regularization = self.output_layer.lam * weight_sum
        return self.compute_loss(P, y) + regularization
    
    def compute_acc(self, P, y):
        """
        Compute accuracy of predictions.

        Args:
            P: Array of shape (K, N) containing the predicted probabilities for each class and sample
            y: Array of shape (N,) containing the integer labels for each sample
    
        Returns:
            acc: The computed accuracy value
        """
        predicted_labels = np.argmax(P, axis=0)
        acc = np.mean(predicted_labels == y)
        return acc
        
    def backward_pass(self, X, Y):
        """
        Backward pass to compute gradients of weights and biases.

        Args:
            X: Array of shape (d, N) containing the input data
            Y: Array of shape (K, N) containing the one-hot-encoded labels
            layer_output: List of arrays holding the output of each layer (after activation)

        Returns:
            gradients: Dictionary containing the gradients of weights and biases for each layer
        """
        G = None
        for i, l in enumerate(reversed(self.layers)):
            # We only pass Y if we're at the output layer
            Y_pass = Y if i == 0 else None
            if l.__class__.__name__ == "ConvolutionalLayer":
                G = l.backward(G)
            elif l.__class__.__name__ == "FullyConnectedLayer":
                G, W, b = l.backward(G, Y=Y_pass)
            elif l.__class__.__name__ == "FlattenLayer":
                G = l.backward(G)

    def mini_batch_gd(
            self,
            rng,
            loss_func,
            cost_func,
            data_handler,
            GDparams,
            cyclical_lr=None, 
            verbose=2,
            record_every=1
        ):
        """
        Mini-batch gradient descent.

        Args:
            rng: Random number generator (passed for reproducibility)
            loss_func: Function to compute the loss (cross-entropy)
            cost_func: Function to compute the cost (loss + regularization)
            data_handler: DataHandler object containing the training and validation data
            GDparams: Dictionary holding the parameters for gradient descent (n_batch, eta, n_epochs)
            cyclical_lr: Dictionary holding parameters for cyclical learning rate (n_s, eta_min, eta_max) or None for no cyclical learning rate
            verbose: Level of verbosity for printing progress and plotting results (0 = no output, 1 = print progress, 2 = print progress and plot results)
            record_every: Number of epochs between recording loss and cost history for plotting
        """
        # And extract parameters
        n_batch = GDparams['n_batch']
        eta = GDparams['eta']
        n_epochs = GDparams['n_epochs']
        n_train = data_handler.X_train.shape[1]
        # And prepare a loss and cost history lists
        training_loss_history = []
        validation_loss_history = []
        training_cost_history = []
        validation_cost_history = []
        training_acc_history = []
        validation_acc_history = []
        # For each epoch
        t = 0
        for _epoch in range(n_epochs):
            # We first shuffle the data
            perm = rng.permutation(n_train)
            X_train_shuffled = data_handler.X_train[:, perm]
            Y_train_shuffled = data_handler.Y_train[:, perm]
            y_train_shuffled = data_handler.y_train[perm]
            # Data augmentation
            #X_train_shuffled = augment_image(X_train_shuffled, prob=0.5)

            # Then we perform per mini-batch updates
            for j in range(n_train//n_batch): # Changed from / to // for integer division
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                X_batch = X_train_shuffled[:, j_start:j_end]
                Y_batch = Y_train_shuffled[:, j_start:j_end]
                #y_batch = y_train_shuffled[j_start:j_end]

                # Cyclical learning rate
                if not cyclical_lr is None:
                    n_s = cyclical_lr['n_s']
                    eta_min = cyclical_lr['eta_min']
                    eta_max = cyclical_lr['eta_max']

                    l = t % (2 * n_s)

                    if cyclical_lr['increasing_step_sizes']:
                        if t > 0 and t % (2 * n_s) == 0:
                            cyclical_lr['n_s'] = n_s * 2
                            n_s = cyclical_lr['n_s']

                    if l < n_s:
                        eta = eta_min + l / n_s * (eta_max - eta_min)
                    else:
                        eta = eta_max - (l - n_s) / n_s * (eta_max - eta_min)
                    t += 1

                # We apply the network and compute the gradients
                self.apply_network(X_batch)
                self.backward_pass(X_batch, Y_batch)
                # Finally we update the weights and biases
                for layer in self.layers:
                    layer.update_params(eta)

            # At the end of the epoch, we compute the training loss
            # and validation loss and store them in the history lists
            # for later comparison
            P_train = self.apply_network(data_handler.X_train)
            train_loss = loss_func(P_train, data_handler.y_train)
            training_loss_history.append(train_loss)
            train_cost = cost_func(P_train, data_handler.y_train)
            training_cost_history.append(train_cost)

            P_val = self.apply_network(data_handler.X_val)
            val_loss = loss_func(P_val, data_handler.y_val)
            validation_loss_history.append(val_loss)
            val_cost = cost_func(P_val, data_handler.y_val)
            validation_cost_history.append(val_cost)

            train_acc = self.compute_acc(P_train, data_handler.y_train)
            training_acc_history.append(train_acc)
            val_acc = self.compute_acc(P_val, data_handler.y_val)
            validation_acc_history.append(val_acc)

            # We print the training and validation loss at the end of each epoch
            # to monitor the process
            if verbose >= 1:
                print(f"Epoch {_epoch+1}/{n_epochs}, Train. Loss: {training_loss_history[-1]:.4f}, Val. Loss: {validation_loss_history[-1]:.4f}, Train. Acc: {training_acc_history[-1]:.4f}, Val. Acc: {validation_acc_history[-1]:.4f}")

        # Before returning, we plot the training and
        # validation loss & cost history using matplotlib

        if verbose >= 2:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            n_batches_per_epoch = data_handler.X_train.shape[1] // GDparams['n_batch']
            updates_recorded = np.arange(len(training_cost_history)) * record_every * n_batches_per_epoch

            ax[1].plot(updates_recorded, training_cost_history, label='Training', color='green')
            ax[1].plot(updates_recorded, validation_cost_history, label='Validation', color='red')
            ax[1].set_xlabel('Update Step')
            ax[1].set_ylabel('Cost')
            ax[1].legend()
            
            ax[0].plot(updates_recorded, training_loss_history, label='Training', color='green')
            ax[0].plot(updates_recorded, validation_loss_history, label='Validation', color='red')
            ax[0].set_xlabel('Update Step')
            ax[0].set_ylabel('Loss')
            ax[0].legend()

            ax[2].plot(updates_recorded, training_acc_history, label='Training', color='green')
            ax[2].plot(updates_recorded, validation_acc_history, label='Validation', color='red')
            ax[2].set_xlabel('Update Step')
            ax[2].set_ylabel('Accuracy')
            ax[2].legend()

            plt.tight_layout()
            plt.show()

        # Store all history lists in the class attributes for later use
        self.training_loss_history = training_loss_history
        self.validation_loss_history = validation_loss_history
        self.training_cost_history = training_cost_history
        self.validation_cost_history = validation_cost_history
        self.training_acc_history = training_acc_history
        self.validation_acc_history = validation_acc_history

    def cyclic_mini_batch_gd(
            self,
            rng,
            loss_func,
            cost_func,
            data_handler,
            GDparams,
            cyclical_lr, 
            verbose=2,
            record_every=1,
            label_smoothing = None
        ):
        """
        Cyclic mini-batch gradient descent.

        Args:
            rng: Random number generator (passed for reproducibility)
            loss_func: Function to compute the loss (cross-entropy)
            cost_func: Function to compute the cost (loss + regularization)
            data_handler: DataHandler object containing the training and validation data
            GDparams: Dictionary holding the parameters for gradient descent (n_batch)
            cyclical_lr: Dictionary holding parameters for cyclical learning rate (n_s, eta_min, eta_max) or None for no cyclical learning rate
            verbose: Level of verbosity for printing progress and plotting results (0 = no output, 1 = print progress, 2 = print progress and plot results)
        """
        # And extract parameters
        n_batch = GDparams['n_batch']
        cycles = cyclical_lr['cycles']
        n_train = data_handler.X_train.shape[1]
        # And prepare a loss and cost history lists
        training_loss_history = []
        validation_loss_history = []
        training_cost_history = []
        validation_cost_history = []
        training_acc_history = []
        validation_acc_history = []
        # 
        t = 0
        t_in_cycle = 0
        cycle = 0
        n_s = cyclical_lr['n_s']
        eta_min = cyclical_lr['eta_min']
        eta_max = cyclical_lr['eta_max']
        increasing_step_sizes = cyclical_lr.get('increasing_step_sizes', False)
        # Label smoothing
        Y_smooth = None
        Y = data_handler.Y_train
        if not label_smoothing is None:
            Y_smooth = np.copy(data_handler.Y_train.T)
            for Y in Y_smooth:
                Y[Y == 1] = 1 - label_smoothing
                Y[Y == 0] = label_smoothing / (Y.shape[0] - 1)
            Y_smooth = Y_smooth.T
            Y = Y_smooth
        
        # For each cycle
        while cycle < cycles:
            # We first shuffle the data
            perm = rng.permutation(n_train)
            X_train_shuffled = data_handler.X_train[:, perm]
            Y_train_shuffled = Y[:, perm]
            #y_train_shuffled = data_handler.y_train[perm]
            # Data augmentation
            #X_train_shuffled = augment_image(X_train_shuffled, prob=0.5)

            # Then we perform per mini-batch updates
            for j in range(n_train//n_batch): # Changed from / to // for integer division
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                X_batch = X_train_shuffled[:, j_start:j_end]
                Y_batch = Y_train_shuffled[:, j_start:j_end]
                #y_batch = y_train_shuffled[j_start:j_end]

                l = t_in_cycle % (2 * n_s)

                eta = None

                if t_in_cycle < n_s:
                    eta = eta_min + l / n_s * (eta_max - eta_min)
                else:
                    eta = eta_max - (l - n_s) / n_s * (eta_max - eta_min)

                t += 1
                t_in_cycle += 1

                if t_in_cycle == 2 * n_s:
                    cycle += 1
                    if cycle >= cycles:
                        break
                    if increasing_step_sizes:
                        n_s *= 2
                    t_in_cycle = 0

                # We apply the network and compute the gradients
                self.apply_network(X_batch)
                self.backward_pass(X_batch, Y_batch)
                # Finally we update the weights and biases
                for layer in self.layers:
                    layer.update_params(eta)

            if cycle >= cycles:
                break

            # At the end of the cycle, we compute the training loss
            # and validation loss and store them in the history lists
            # for later comparison
            P_train = self.apply_network(data_handler.X_train)
            train_loss = loss_func(P_train, data_handler.y_train)
            training_loss_history.append(train_loss)
            train_cost = cost_func(P_train, data_handler.y_train)
            training_cost_history.append(train_cost)

            P_val = self.apply_network(data_handler.X_val)
            val_loss = loss_func(P_val, data_handler.y_val)
            validation_loss_history.append(val_loss)
            val_cost = cost_func(P_val, data_handler.y_val)
            validation_cost_history.append(val_cost)

            train_acc = self.compute_acc(P_train, data_handler.y_train)
            training_acc_history.append(train_acc)
            val_acc = self.compute_acc(P_val, data_handler.y_val)
            validation_acc_history.append(val_acc)

            # We print the training and validation loss at the end of each update
            # to monitor the process
            if verbose >= 1:
                print(f"Cycle {cycle+1}/{cycles}, Train. Loss: {training_loss_history[-1]:.4f}, Val. Loss: {validation_loss_history[-1]:.4f}, Train. Acc: {training_acc_history[-1]:.4f}, Val. Acc: {validation_acc_history[-1]:.4f}")

        # Before returning, we plot the training and
        # validation loss & cost history using matplotlib

        if verbose >= 2:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            n_batches_per_ = data_handler.X_train.shape[1] // GDparams['n_batch']
            updates_recorded = np.arange(len(training_cost_history)) * record_every * n_batches_per_

            ax[1].plot(updates_recorded, training_cost_history, label='Training', color='green')
            ax[1].plot(updates_recorded, validation_cost_history, label='Validation', color='red')
            ax[1].set_xlabel('Update Step')
            ax[1].set_ylabel('Cost')
            ax[1].legend()
            
            ax[0].plot(updates_recorded, training_loss_history, label='Training', color='green')
            ax[0].plot(updates_recorded, validation_loss_history, label='Validation', color='red')
            ax[0].set_xlabel('Update Step')
            ax[0].set_ylabel('Loss')
            ax[0].legend()

            ax[2].plot(updates_recorded, training_acc_history, label='Training', color='green')
            ax[2].plot(updates_recorded, validation_acc_history, label='Validation', color='red')
            ax[2].set_xlabel('Update Step')
            ax[2].set_ylabel('Accuracy')
            ax[2].legend()

            plt.tight_layout()
            plt.show()

        # Store all history lists in the class attributes for later use
        self.training_loss_history = training_loss_history
        self.validation_loss_history = validation_loss_history
        self.training_cost_history = training_cost_history
        self.validation_cost_history = validation_cost_history
        self.training_acc_history = training_acc_history
        self.validation_acc_history = validation_acc_history
