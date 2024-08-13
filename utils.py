import numpy as np
import pandas as pd




def serve_data(X, window_length):
    """
    Apply sliding window and reshape output
    """
    # Apply sliding window
    X_ = np.lib.stride_tricks.sliding_window_view(X, window_shape = window_length, axis=0)

    # Reshape into (windows, features*lags)
    X_ = X_.reshape(X_.shape[0], -1)

    return X_
