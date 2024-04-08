import numpy as np
from tqdm import tqdm

# Original implementation. But modified for multi-channel data.
# https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py


def jitter(x, sigma=0.03):
    # Supported
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1, columns=None):
    """
    Apply scaling to specified columns of a (T, N) dataset.

    Parameters:
    - x: Input data of shape (T, N), where T is time and N is the number of features.
    - sigma: Standard deviation of the Gaussian distribution used for generating the scaling factor.
    - columns: A list of column indices to apply the scaling. If None, applies to all columns.

    Returns:
    - Scaled data with the same shape as the input.
    """
    if columns is None:
        # Apply scaling to all columns
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[1]))
        print(factor)
    else:
        # Apply scaling only to specified columns
        factor = np.ones((x.shape[0], x.shape[1]))  # Initialize factors as 1 for all elements
        scaling_values = np.random.normal(loc=1., scale=sigma, size=(len(columns)))
        factor[:, columns] = scaling_values  # Replace factors for specified columns

    return np.multiply(x, factor)


def rotate(x, rotation_range):
    from scipy.spatial.transform import Rotation as R
    """
    Apply the same random rotation to all 3D data points sets within each sample.

    Parameters:
    - x: Input data of shape (N, 3*M), where N is the number of samples and
         3*M represents multiple sets of 3D points.
    - rotation_range: A dictionary specifying the range of rotation (in degrees) for each axis:
                      {'x': (min_angle, max_angle),
                       'y': (min_angle, max_angle),
                       'z': (min_angle, max_angle)}
    - M: The number of sets of 3D points per sample (each set has 3 dimensions).

    Returns:
    - Rotated data with the same shape as the input.
    """
    # Randomly pick a rotation angle within the specified range for each axis
    angles = {axis: np.random.uniform(*rotation_range[axis]) for axis in ['x', 'y', 'z']}
    print(angles)
    # Convert angles from degrees to radians
    angles_rad = np.deg2rad([angles['x'], angles['y'], angles['z']])

    # Create a rotation object from Euler angles
    rotation = R.from_euler('xyz', angles_rad)

    # Initialize the rotated data array
    rotated_x = np.zeros_like(x)
    M = x.shape[1] // 3
    # Apply the same rotation to each set of 3D points across all samples
    for i in range(M):
        # Extract the ith set of 3D points across all samples
        points_set = x[:, 3 * i:3 * (i + 1)]

        # Apply rotation
        rotated_set = rotation.apply(points_set)

        # Place the rotated set-back into the corresponding columns of the output array
        rotated_x[:, 3 * i:3 * (i + 1)] = rotated_set

    return rotated_x


def permutation(x, max_segments=5, seg_mode="equal"):
    # Supported
    orig_steps = np.arange(x.shape[0])
    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[0] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        np.random.shuffle(splits)
        permuted_idx = np.concatenate(splits).ravel()
        ret = x[permuted_idx]
    else:
        ret = x
    return ret


def magnitude_warp(x, sigma=0.2, knot=4, columns=None):
    # Supported
    from scipy.interpolate import CubicSpline
    # Ensure orig_steps is compatible with the (T, N) shape
    orig_steps = np.arange(x.shape[0])

    # Generate a single random warp for all features
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))

    # Generate warp steps, ensuring it's based on the time dimension
    warp_steps = np.linspace(0, x.shape[0] - 1., num=knot + 2)

    # Create the warping curve using CubicSpline, applied to the time dimension
    warper = CubicSpline(warp_steps, random_warps)(orig_steps)

    # Apply the same warping curve across all features
    # Apply the warping curve
    if columns is None:
        # If no columns specified, apply to all columns
        ret = x * warper[:, None]
    else:
        # Apply warping only to specified columns
        ret = x.copy()  # Create a copy to leave the original array unchanged
        for col in columns:
            ret[:, col] = x[:, col] * warper

    return ret


def time_warp(x, sigma=0.2, knot=4, columns=None):
    # Supported
    from scipy.interpolate import CubicSpline
    """
    Apply time warping to specified columns of a (T, N) dataset.

    Parameters:
    - x: Input data of shape (T, N), where T is time and N is the number of features.
    - sigma: Standard deviation of the Gaussian distribution used for generating the warp.
    - knot: Number of knots used in the cubic spline, excluding the endpoints.
    - columns: A list of column indices to apply the warping. If None, applies to all columns.

    Returns:
    - Warped data with the same shape as the input.
    """
    orig_steps = np.arange(x.shape[0])
    if columns is None:
        # Apply warping to all columns if none specified
        columns = range(x.shape[1])

    # Generate warping parameters
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    warp_steps = np.linspace(0, x.shape[0] - 1., num=knot + 2)

    ret = x.copy()  # Work on a copy of x to retain original data
    for col in columns:
        # Generate time warp for each specified column
        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
        scale = (x.shape[0] - 1) / time_warp[-1]
        ret[:, col] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1), x[:, col])

    return ret

