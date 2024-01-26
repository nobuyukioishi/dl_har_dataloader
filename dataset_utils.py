import numpy as np


# def sliding_window(x, y, window, stride, scheme="last"):
def sliding_window(x, y, window, stride, scheme="max"):
    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target


def standardize(data):
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - np.mean(data, axis=0)) / std


def normalize(data: np.ndarray, min_vals: np.ndarray | None = None, max_vals: np.ndarray | None = None, verbose=False):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param min_vals: numpy integer array
        Array containing the minimum values for each sensor channel
    :param max_vals: numpy integer array
        Array containing the maximum values for each sensor channel
    :param verbose: bool
        Whether to print debug information
    :return:
        Normalized sensor data
    """
    if min_vals is None:
        min_vals = np.min(data, axis=0)
    if max_vals is None:
        max_vals = np.max(data, axis=0)

    assert min_vals.shape == max_vals.shape == data.shape[1:], \
        f'Shape mismatched! min_vals: {min_vals.shape}, max_vals: {max_vals.shape}, data: {data.shape}'
    assert np.all(min_vals <= max_vals), f'min_vals: {min_vals}, max_vals: {max_vals}'

    # Avoid division by zero in case max_val equals min_val
    range_vals = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
    normalized_data = (data - min_vals) / range_vals

    if verbose:
        print(f'min_vals: {min_vals}')
        print(f'max_vals: {max_vals}')
        print(f'range_vals {range_vals}')
        print(f'normalized_data - max: {np.max(normalized_data, axis=0)}')
        print(f'normalized_data - min: {np.min(normalized_data, axis=0)}')

    return normalized_data
