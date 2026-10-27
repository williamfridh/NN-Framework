import numpy as np

# Prepare global variables
DATASET_DIR = "../dataset/"     # Location of dataset

class DataHandler:
    """
    Class for data management, including loading, normalization and augmentation.

    Attributes:
        X_train: Array of shape (d, N_train) containing the training image data
        X_val: Array of shape (d, N_val) containing the validation image data
        X_test: Array of shape (d, N_test) containing the test image data

        Y_train: Array of shape (K, N_train) containing the one-hot-encoded training labels
        Y_val: Array of shape (K, N_val) containing the one-hot-encoded validation labels
        Y_test: Array of shape (K, N_test) containing the one-hot-encoded test labels

        y_train: Array of shape (N_train,) containing the integer training labels
        y_val: Array of shape (N_val,) containing the integer validation labels
        y_test: Array of shape (N_test,) containing the integer test labels

        d: Number of features (input dimension)
        K: Number of classes
        x_shape: Shape of images (e.g. (32, 32, 3) for CIFAR-10)
    """

    def __init__(self, train_files=None, val_files=None, test_files=None, normalize=True, d=None, k=None, x_shape=None):

        if train_files is None and val_files is None and test_files is None:
            raise ValueError("No files provided for Data initialization. Please provide at least one of train_files, val_files or test_files.")
        if d is None or k is None or x_shape is None:
            raise ValueError("Missing required parameters for Data initialization. Please provide arguments d, k and x_shape.")

        self.d = d
        self.K = k
        self.x_shape = x_shape

        if train_files is not None and len(train_files) > 0:
            self.X_train, self.Y_train, self.y_train = self.load_data_files(train_files)
        else:
            self.X_train, self.Y_train, self.y_train = [], [], []

        if val_files is not None and len(val_files) > 0:
            self.X_val, self.Y_val, self.y_val = self.load_data_files(val_files)
        else:
            self.X_val, self.Y_val, self.y_val = [], [], []

        if test_files is not None and len(test_files) > 0:
            self.X_test, self.Y_test, self.y_test = self.load_data_files(test_files)
        else:
            self.X_test, self.Y_test, self.y_test = [], [], []

        if normalize:
            self.normalize()

        self.n_p = (self.x_shape[0] // 4) ** 2
        self.n_f = 2

    def load_data_files(self, batch_file_list):
        """
        Loads single or multiple data files.

        Args:
            batch_file_list: A list of batch file names to load (e.g. ["data_batch_1", "data_batch_2"])

        Returns:
            X: Array of shape (d, N) containing the image data (N is total number of images)
            Y: Array of shape (K, N) containing the one-hot-encoded labels
            y: Array of shape (N,) containing the integer labels
        """
        X = []
        Y = []
        y = []
        for batch_file in batch_file_list:
            with open(DATASET_DIR + batch_file, "rb") as f:
                batch = np.load(f, allow_pickle=True, encoding="bytes")
            # Convert images to float
            X.append(batch[b"data"].astype(float).T)
            # Grab one-hot-encoded labels
            Y.append(np.zeros((self.K, X[-1].shape[1])))
            for i, label in enumerate(batch[b"labels"]):
                Y[-1][label, i] = 1
            # Grab labels
            y.append(np.array(batch[b"labels"], dtype=np.int64))
        # Flatten the lists into arrays
        X = np.concatenate(X, axis=1)
        Y = np.concatenate(Y, axis=1)
        y = np.concatenate(y, axis=0)
        print(f"Loaded total of {np.array(X).shape[1]} samples:")
        print(f"\tX shape: {np.array(X).shape}")
        print(f"\tY shape: {np.array(Y).shape}")
        print(f"\ty shape: {np.array(y).shape}\n")
        return X, Y, y
    
    def augment_image(self, X, prob):
        """
        Args:
            X: The input data, a 2D array of shape (d, N), where N is the number of images
            prob: The probability of performing each augmentation (horizontal flip and rotation)

        Returns:
            The augmented data same shape as the input
        """
        # We reshape the data into images (as done earlier)
        X = X.reshape(*self.x_shape, -1, order="F")
        X = np.transpose(X, (1, 0, 2, 3))
        # For each image, we perform data augmentation with a probability of prob
        for i in range(X.shape[3]):
            # We  add a chanse of horizontal flip
            if np.random.rand() < prob:
                X[:, :, :, i] = np.fliplr(X[:, :, :, i])
            # and check if we should rotate or not
            if np.random.rand() < prob:
                # We rotate the image by 90 degrees
                X[:, :, :, i] = np.rot90(X[:, :, :, i], k=1)
        # After augmentation we restore the original shape
        X = np.transpose(X, (1, 0, 2, 3))
        X = X.reshape(self.d, -1, order="F")

        return X
    
    @staticmethod
    def split_data(X, Y, y, train_size=0.8, val_size=0.1):
        """
        Splits data into training, validation and testing sets.

        Note that test_size is calculated as 1 - train_size - val_size.

        Args:
            X: Array of shape (:, N) containing the image data
            Y: Array of shape (:, N) containing the one-hot-encoded labels
            y: Array of shape (N,) containing the integer labels
            train_size: Proportion of training set
            val_size: Proportion of validation set

        Returns:
            X_train, Y_train, y_train: Training data and labels
            X_val, Y_val, y_val: Validation data and labels
            X_test, Y_test, y_test: Testing data and labels
        """
        n = X.shape[1]

        X_train = X[:, :int(train_size * n)]
        Y_train = Y[:, :int(train_size * n)]
        y_train = y[:int(train_size * n)]

        X_val = X[:, int(train_size * n):int((train_size + val_size) * n)]
        Y_val = Y[:, int(train_size * n):int((train_size + val_size) * n)]
        y_val = y[int(train_size * n):int((train_size + val_size) * n)]

        X_test = X[:, int((train_size + val_size) * n):]
        Y_test = Y[:, int((train_size + val_size) * n):]
        y_test = y[int((train_size + val_size) * n):]

        return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test
    
    def normalize(self):
        """
        Calculated mean standard deviation of training data and normalizes the data.
        This includes training, validation and test data.
        """
        mean = np.mean(self.X_train, axis=1).reshape(-1, 1)
        std = np.std(self.X_train, axis=1).reshape(-1, 1)
        self.X_train = (self.X_train - mean) / std
        self.X_val = (self.X_val - mean) / std
        self.X_test = (self.X_test - mean) / std
