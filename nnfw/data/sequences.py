import gc
import os

import numpy as np

class DataHandler:
    """
    Note:
    Do not replace NumPy with CuPy, as the data loading and preprocessing is done on CPU,
    and only the batches are sent to GPU for training.
    """

    def __init__(self, folder_path, seq_length, seq_step, look_ahead=1, batch_size=32):
        self.folder_path = folder_path
        self.seq_length = seq_length
        self.seq_step = seq_step
        self.look_ahead = look_ahead
        self.batch_size = batch_size

    def get_steps_per_epoch(self):
        files = sorted([os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.npy')])
        
        total_data_points = 0
        
        for file_path in files:
            data = np.load(file_path, mmap_mode='r')
            total_ticks = data.shape[0] - 1
            last_possible_start = total_ticks - self.seq_length - self.look_ahead
            
            # Safely count data points without discarding remainders
            if last_possible_start > 0:
                total_data_points += len(range(0, last_possible_start, self.seq_step))
                
            del data
            
        return total_data_points // self.batch_size
    
    def get_generator(self):
        files = sorted([os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.npy')])
        label_idx = [2, 1047]

        # Initialize ourside the loop so remainders are carried across files
        X_batch = []
        y_batch = []

        for file_path in files:
            data = np.load(file_path, mmap_mode='r')
            
            # GUARD: Prevent thread crash from malformed files
            if data.shape[1] <= max(label_idx):
                print(f"\n[WARNING] Skipping {file_path}: shape {data.shape} is too small!")
                del data
                continue
                
            total_ticks = data.shape[0] - 1
            last_possible_start = total_ticks - self.seq_length - self.look_ahead

            if last_possible_start > 0:
                for i in range(0, last_possible_start, self.seq_step):
                    X_seq = np.copy(data[i:i + self.seq_length])
                    label_tick = i + self.seq_length + self.look_ahead - 1
                    y_label = np.copy(data[label_tick, label_idx])

                    X_batch.append(X_seq.T)
                    y_batch.append(y_label)

                    if len(X_batch) == self.batch_size:
                        yield np.stack(X_batch, axis=1), np.stack(y_batch, axis=1)
                        X_batch = []
                        y_batch = []
            
            del data
            
        gc.collect()
