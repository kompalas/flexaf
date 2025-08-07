import logging
import logging.config
import re
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import random
import argparse
import tensorflow
import yaml
from collections import Counter
from enum import Enum
from keras.api.utils import to_categorical
from sklearn.metrics import f1_score
from datetime import datetime
from glob import glob
from src import project_dir
from src.args import cmd_args, validate_args


__all__ = [
    'env_cfg', 'cfg_from_yaml', 'logging_cfg', 'get_timestamp', 
    'split_dataset', 'best_split', 'gini_impurity',
    'add_noise', 'add_relative_noise', 'stochastic_rounding',
    'resample', 'resample_dataframe', 'normalization', 'convert_to_fixed_point',
    'accuracy_keras', 'f1score_keras', 'transform_categorical',
    'get_size', 'clear_directory', 'colored',
]

logger = logging.getLogger(__name__)


def env_cfg():
    """Configure the environment to run the genetic algorithm"""
    parser = argparse.ArgumentParser("Feature Selection for FE - Genetic Algorithm")
    parser = cmd_args(parser)
    args = parser.parse_args()

    # overwrite if a yaml configuration file is given
    assert args.yaml_cfg_file is not None, "Specify a yaml file to configure the experiment"
    cfg_from_yaml(args, args.yaml_cfg_file)

    # check for errors
    validate_args(args)

    if args.deterministic:
        np.random.seed(args.global_seed)
        random.seed(args.global_seed)
        tensorflow.random.set_seed(args.global_seed)

    # configure logging 
    logging_cfg(args)

    # fix deepcopy recursion problem by increasing the limit, if necessary
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)

    return args


def cfg_from_yaml(args, cfg_yaml_file):
    """Configure environment based on arguments from a yaml file
    """
    def replace_arg(name, value):
        # special handling for Enum type of arguments
        if isinstance(getattr(args, name, None), Enum) and value is not None:
            assert isinstance(value, str)
            value = next(entry.value for entry in getattr(args, name).__class__
                         if entry.name.lower() == value)
            value = getattr(args, name).__class__(value)
        # special handling for lists (nargs='+/*/?')
        elif isinstance(getattr(args, name, None), list) and value is not None:
            value = [value] if not isinstance(value, list) else value
        setattr(args, name, value)

    # read configuration file
    with open(cfg_yaml_file, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)

    # inspect all arguments
    for name, value in yaml_dict.items():
        # we assume a two-level nested dictionary
        if isinstance(value, dict):
            # usually this branch gets executed
            for _name, _value in value.items():
                replace_arg(_name, _value)
        else:
            # this is rarely executed
            replace_arg(_name, _value)


def logging_cfg(args):
    """Configure logging for entire framework"""
    if not os.path.exists(os.path.join(project_dir, 'logs')):
        os.makedirs(os.path.join(project_dir, 'logs'))

    # set the name of the log file and directory
    timestr = get_timestamp()
    exp_full_name = timestr if args.name is None else args.name + '___' + timestr
    logdir = os.path.join(project_dir, 'logs', exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # use the logging config file
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    logging.config.fileConfig(
        os.path.join(project_dir, 'logging.conf'),
        disable_existing_loggers=False,
        defaults={
            'main_log_filename': f'{project_dir}/logs/out.log',
            'all_log_filename': log_filename,
        }
    )

    # initialized logger and first messages
    logging.getLogger().logdir = logdir
    logger.log_filename = log_filename
    logger.info('Log file for this run: ' + os.path.realpath(log_filename))
    logger.debug("Command line: {}".format(" ".join(sys.argv)))
    arguments = {argument: getattr(args, argument) for argument in dir(args)
                 if not callable(getattr(args, argument)) and not argument.startswith('__')}
    logger.debug(f"Arguments: {arguments}")

    # Create a symbollic link to the last log file created (for easier access)
    try:
        os.unlink("latest_log_file")
    except FileNotFoundError:
        pass
    try:
        os.unlink("latest_log_dir")
    except FileNotFoundError:
        pass
    try:
        os.symlink(logdir, "latest_log_dir")
        os.symlink(log_filename, "latest_log_file")
    except OSError:
        logger.debug("Failed to create symlinks to latest logs")


def get_timestamp():
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")[:-3]


def split_dataset(X, y, feature_index, threshold):
    """Split dataset into left/right based on a threshold for a feature."""
    left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]
    
    X_left = [X[i] for i in left_indices]
    y_left = [y[i] for i in left_indices]
    X_right = [X[i] for i in right_indices]
    y_right = [y[i] for i in right_indices]
    
    return X_left, y_left, X_right, y_right


def best_split(X, y, features_to_consider):
    """Find the best feature and threshold to split on among allowed features."""
    best_feature = None
    best_threshold = None
    best_impurity = float('inf')
    best_split_data = None
    
    n_features = len(X[0])
    
    for feature_index in features_to_consider:
        # Get sorted unique values in this column
        feature_values = [row[feature_index] for row in X]
        thresholds = sorted(set(feature_values))
        
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            if not y_left or not y_right:
                continue  # skip empty splits
            
            impurity_left = gini_impurity(y_left)
            impurity_right = gini_impurity(y_right)
            total = len(y_left) + len(y_right)
            weighted_impurity = (len(y_left) / total) * impurity_left + (len(y_right) / total) * impurity_right
            
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_feature = feature_index
                best_threshold = threshold
                best_split_data = (X_left, y_left, X_right, y_right)
    
    return best_feature, best_threshold, best_split_data


def gini_impurity(y):
    """Compute Gini impurity for a list of class labels y."""
    counts = Counter(y)
    total = len(y)
    if total == 0:
        return 0
    probabilities = [count / total for count in counts.values()]
    return 1 - sum(p ** 2 for p in probabilities)


def add_noise(data: np.ndarray, noise_level=0.01, distribution='normal') -> np.ndarray:
    """
    Add random noise to a NumPy array of any shape.

    Parameters:
    - data (np.ndarray): Original data array.
    - noise_level (float): Standard deviation (normal) or range limit (uniform).
    - distribution (str): 'normal' (Gaussian) or 'uniform'.

    Returns:
    - np.ndarray: Noisy version of the input data.
    """
    if distribution == 'normal':
        noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    elif distribution == 'uniform':
        noise = np.random.uniform(low=-noise_level, high=noise_level, size=data.shape)
    else:
        raise ValueError("Unsupported distribution. Choose 'normal' or 'uniform'.")

    return data + noise


def add_relative_noise(data: np.ndarray, noise_percent=0.1, distribution='uniform') -> np.ndarray:
    """
    Add value-dependent (multiplicative) noise to simulate sensor digitization effects (e.g. ADC noise).

    Parameters:
    - data (np.ndarray): Input array of sensor readings or any measurements.
    - noise_percent (float): Relative noise level (e.g. 0.1 means ±10%).
    - distribution (str): Type of distribution ('uniform' or 'normal').

    Returns:
    - np.ndarray: Noisy array where noise scales with the data.
    """
    if distribution == 'uniform':
        noise_factor = np.random.uniform(low=1 - noise_percent,
                                         high=1 + noise_percent,
                                         size=data.shape)
    elif distribution == 'normal':
        noise_factor = np.random.normal(loc=1.0,
                                        scale=noise_percent,
                                        size=data.shape)
    else:
        raise ValueError("Unsupported distribution. Use 'uniform' or 'normal'.")

    return data * noise_factor


def resample(data, current_sampling_rate, new_sampling_rate):
    """Resample data to a given re-sampling rate"""
    if current_sampling_rate == new_sampling_rate:
        return data

    if new_sampling_rate < current_sampling_rate:
        interval = int(current_sampling_rate / new_sampling_rate)
        # Downsampling: take values at specific intervals
        resampled_data = data[::interval]
    else:
        # Upsampling: use sample and hold
        factor = int(new_sampling_rate / current_sampling_rate)
        resampled_data = np.repeat(data, factor)[:int(len(data) * factor)]

    return resampled_data

def resample_dataframe(df, current_sampling_rate, new_sampling_rate):
    """Resample a DataFrame to a given re-sampling rate"""
    if current_sampling_rate == new_sampling_rate:
        return df

    resampled_df = pd.DataFrame()
    # resampled_df = df.copy()
    for column in df.columns:
        new_data = resample(df[column].values, current_sampling_rate, new_sampling_rate)
        resampled_df[column] = new_data

    return resampled_df


def normalization(data):
    """Min-max normalization of a given data"""
    if isinstance(data, pd.Series):
        data = data.values
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data) == 0:
        return data

    min_val = np.min(data)
    max_val = np.max(data)

    if min_val == max_val:
        return np.zeros_like(data)  # Prevent division by zero

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def convert_to_fixed_point(data, precision, normalize=None, rescale=False, signed=True, return_type=int, fractional_bits=None):
    """Convert data to fixed-point representation with normalization.
    """
    if precision == 32:
        return data

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if fractional_bits is None:
        fractional_bits = {
            3: 2,  # Q0.2
            4: 3,  # Q0.3
            5: 4,  # Q0.4
            6: 3,  # Q2.3
            7: 4,  # Q2.4
            8: 5,  # Q2.5
        }.get(precision, 0)  # default to 0 fractional bits

    min_val, max_val = np.min(data), np.max(data)
    sign_bit = 1 if signed or normalize == '-1->1' else 0
    if normalize == '0->1':
        if min_val != max_val:
            # Normalize the data to fit within 0 to 1 range
            data = (data - min_val) / (max_val - min_val)
    elif normalize == '-1->1':
        signed = True
        sign_bit = 1  # Set sign bit for signed representation
        if min_val == max_val:
            data = np.zeros_like(data)  # Prevent division by zero
        else:
            data = 2 * (data - min_val) / (max_val - min_val) - 1
    elif normalize is not None:
        raise ValueError(f"Unsupported normalization type ({normalize}): use '0->1' or '-1->1'")

    # Scale to fixed-point range and clip
    factor = 2 ** fractional_bits
    scaled_data = np.round(data * factor).astype(int)
    if signed:
        max_fixed_point, min_fixed_point = (2 ** (precision - sign_bit)) - 1, -(2 ** (precision - sign_bit))
    else:
        max_fixed_point, min_fixed_point = (2 ** precision) - 1, 0
    clipped_data = np.clip(scaled_data, min_fixed_point, max_fixed_point)
    if rescale:
        clipped_data = clipped_data / factor
        return clipped_data.astype(float)
    return clipped_data.astype(return_type)


def stochastic_rounding(input_array):
    sign = np.sign(input_array)  # Get the sign (-1, 0, or 1)
    stochastic_matrix = np.random.rand(*input_array.shape)  # Generate random values in [0,1)
    output_array = np.floor(np.abs(input_array) + stochastic_matrix) * sign  # Apply stochastic rounding
    return output_array


def get_size(directory):
    """Get the size of a directory"""
    # check if it exists
    assert os.path.isdir(directory), f"Directory {directory} doesn't exist."

    # use linux 'du' command to get the directory size in bytes
    cmd = f"du -s {directory}"
    if sys.version_info.minor == 8:
        p = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    elif sys.version_info.minor == 6:
        p = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

    size = re.search(r"(^\d+).*", p.stdout).group(1)
    return int(size)


def accuracy_keras(y_test, y_pred, **accuracy_params):
    """Get the accuracy of a Keras model"""
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return np.mean(y_test == y_pred)


def f1score_keras(y_test, y_pred, **accuracy_params):
    """Get the F1 score of a Keras model"""
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return f1_score(y_test, y_pred, **accuracy_params)


def transform_categorical(data, num_classes):
    data_array = np.array(data)
    try:
        categ_data = to_categorical((data_array - data_array.min()), num_classes=num_classes)  # avoids classes indexed from 1
    except IndexError:
        class_map = {class_index: new_index for new_index, class_index in enumerate(sorted(np.unique(data_array)))}
        vec_data = np.vectorize(class_map.get)(data_array)
        categ_data = to_categorical(vec_data, num_classes=num_classes)
    return categ_data


def clear_directory(directory, pattern='*'):
    """Clear a given directory of files matching a pattern (default is all files)"""
    contains = glob(f"{directory}/{pattern}")
    for file_or_dir in contains:
        if os.path.isdir(file_or_dir):
            os.rmdir(file_or_dir)
        elif os.path.isfile(file_or_dir):
            os.remove(file_or_dir)


def colored(color='blue'):
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m' # orange on some systems
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[97m'

    RESET = '\033[0m' # called to return to standard terminal text color

    return {
        'black': BLACK,
        'red': RED,
        'green': GREEN,
        'yellow': YELLOW,
        'blue': BLUE,
        'magenta': MAGENTA,
        'cyan': CYAN,
        'light_gray': LIGHT_GRAY,
        'dark_gray': DARK_GRAY,
        'bright_red': BRIGHT_RED,
        'bright_green': BRIGHT_GREEN,
        'bright_yellow': BRIGHT_YELLOW,
        'bright_blue': BRIGHT_BLUE,
        'bright_magenta': BRIGHT_MAGENTA,
        'bright_cyan': BRIGHT_CYAN,
        'white': WHITE,
        'reset': RESET
    }[color]

