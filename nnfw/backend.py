# Array library for the recurrent network.
# Set USE_GPU to True inside a ROCm container with CuPy installed.
# Set USE_GPU to False to run the same code with NumPy.

USE_GPU = False

if USE_GPU:
    import cupy as cp
else:
    import numpy as cp
