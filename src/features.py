import logging
import logging.config
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.fft as scipy_fft
from scipy.signal import find_peaks
from src.utils import resample, convert_to_fixed_point


__all__ = [
    'create_features_data_from_df', 'extract_feature',
    'min', 'max', 'mean', 'std', 'sum', 'skewness', 'kurtosis', 'range', 'stretch', 'median',
    'peaks', 'npeaks', 'muPeaks', 'sumPeaks', 'stdPeaks', 'fft'
]

logger = logging.getLogger(__name__)


def create_features_data_from_df(data, features_dict, inputs_precisions, sampling_rates=[], original_sampling_rate=64, window_size=1, target_clock=1, **kwargs):
    """Create the dataset with the selected features
    """
    # convert the raw data to FXP depending on the ADC precision
    features_data = pd.DataFrame()
    for sensor_id, sensor_features in features_dict.items():
        sensor_type = data.columns[sensor_id]
        precision = inputs_precisions[sensor_id]
        sampling_rate = sampling_rates[sensor_id]

        fxp_data = convert_to_fixed_point(data[sensor_type], precision, normalize=None, rescale=True, signed=False, fractional_bits=precision)
        resampled_data = resample(fxp_data, original_sampling_rate, sampling_rate)
        resampled_labels = resample(data['label'], original_sampling_rate, sampling_rate)

        window_shape = window_size * sampling_rate
        step = target_clock * sampling_rate
        reshaped_data = np.lib.stride_tricks.sliding_window_view(resampled_data, window_shape=window_shape)[::step]
        reshaped_labels = np.lib.stride_tricks.sliding_window_view(resampled_labels, window_shape=window_shape)[::step]

        for feature in sensor_features:
            # extract the given feature
            this_feature_data = extract_feature(reshaped_data, feature, **kwargs)
            # keep the most common label within the window
            labels = stats.mode(reshaped_labels, axis=1)[0].flatten()
            # convert the extracted feature output to FXP depending on the ADC precision
            this_feature_data = convert_to_fixed_point(this_feature_data, precision, normalize='0->1', rescale=True, signed=False, fractional_bits=precision)

            # make sure that the shape of the array with the concatenated features is preserved and no NaNs are inserted
            if features_data.shape[0] > 0:
                this_feature_data = np.resize(this_feature_data, (features_data.shape[0], ))
                labels = np.resize(labels, (features_data.shape[0], ))

            this_feature_df = pd.DataFrame(this_feature_data, columns=[f"{sensor_type}_{feature}"])
            features_data = pd.concat([features_data, this_feature_df], axis=1)

    return features_data, labels


def create_features_data_from_array(data, labels, features_dict, inputs_precisions, sampling_rates=[], original_sampling_rate=64, window_size=1, target_clock=1, **kwargs):
    """Create the dataset with the selected features
    """
    features_data = []
    for sensor_id, sensor_features in features_dict.items():
        precision = inputs_precisions[sensor_id]
        sampling_rate = sampling_rates[sensor_id]

        fxp_data = convert_to_fixed_point(data[:, sensor_id], precision, normalize=None, rescale=True, signed=False, fractional_bits=precision)
        resampled_data = resample(fxp_data, original_sampling_rate, sampling_rate)
        resampled_labels = resample(labels, original_sampling_rate, sampling_rate)

        window_shape = window_size * sampling_rate
        step = target_clock * sampling_rate
        reshaped_data = np.lib.stride_tricks.sliding_window_view(resampled_data, window_shape=window_shape)[::step]
        reshaped_labels = np.lib.stride_tricks.sliding_window_view(resampled_labels, window_shape=window_shape)[::step]

        for feature in sensor_features:
            # extract the given feature
            this_feature_data = extract_feature(reshaped_data, feature, **kwargs)
            # keep the most common label within the window
            filtered_labels = stats.mode(reshaped_labels, axis=1)[0].flatten()
            # convert the extracted feature output to FXP depending on the ADC precision
            this_feature_data = convert_to_fixed_point(this_feature_data, precision, normalize='0->1', rescale=True, signed=False, fractional_bits=precision)

            # make sure that the shape of the array with the concatenated features is preserved and no NaNs are inserted
            if len(features_data) > 0:
                this_feature_data = np.resize(this_feature_data, (features_data[0].shape[0], ))
                filtered_labels = np.resize(filtered_labels, (features_data[0].shape[0], ))

            features_data.append(this_feature_data)

    features_data = np.column_stack(features_data)
    return features_data, filtered_labels


### Feature extraction functions

def extract_feature(data, feature, simplified=False, **kwargs):
    """Extract a feature from a given sensor data"""
    suffix = ''
    if simplified and f'{feature}_feature_simplified' in globals():
        suffix = '_simplified'
    feature = globals()[feature + '_feature' + suffix](data, axis=1)
    return np.nan_to_num(feature)  # Replace NaNs with zeros

def min_feature(data, axis=None):
    return np.min(data, axis=axis)

def max_feature(data, axis=None):
    return np.max(data, axis=axis)

def mean_feature(data, axis=None):
    return np.mean(data, axis=axis)

def std_feature(data, axis=None):
    return np.std(data, axis=axis)

def std_feature_simplified(data, axis=None):
    # Step 1: Calculate the average of squares
    avg_of_squares = np.mean(data ** 2, axis=axis)
    # Step 2: Calculate the square of the average
    square_of_avg = np.mean(data, axis=axis) ** 2
    # Step 3: Compute the difference and take the square root
    std_result = np.sqrt(avg_of_squares - square_of_avg)
    return std_result


def sum_feature(data, axis=None):
    return np.sum(data, axis=axis)

def skewness_feature(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return stats.skew(data, axis=axis)

def skewness_feature_simplified(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.mean(data, axis=axis, keepdims=True)
        m2 = np.mean((data - mean) ** 2, axis=axis)  # Variance (second central moment)
        m3 = np.mean((data - mean) ** 3, axis=axis)  # Third central moment
        return m3 / (m2 ** 1.5)

def kurtosis_feature(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return stats.kurtosis(data, axis=axis, fisher=False)

def kurtosis_feature_simplified(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.mean(data, axis=axis, keepdims=True)
        m2 = np.mean((data - mean) ** 2, axis=axis)  # Variance (second central moment)
        m4 = np.mean((data - mean) ** 4, axis=axis)  # Fourth central moment
        return m4 / (m2 ** 2)
 
def range_feature(data, axis=None):
    return np.ptp(data, axis=axis)  # Peak-to-peak range (max - min)

def stretch_feature(data, axis=None):
    return np.max(np.abs(np.diff(data, axis=axis)), axis=axis)  # Maximum change between adjacent values

def median_feature(data, axis=None):
    return np.median(data, axis=axis)

def peaks_feature(data, axis=0, height=None, distance=None):
    def peak_finder(arr):
        peak_indices, _ = find_peaks(arr, height=height, distance=arr.size)
        return arr[peak_indices] if peak_indices.size > 0 else np.array([arr[np.argmax(arr)]])
    return np.apply_along_axis(peak_finder, axis, data)

# def _peaks_feature(data, axis=0, height=None, distance=None):
#     peaks_data = []
#     for i in range(data.shape[1-axis]):
#         row_or_col = data.take(i, axis=1-axis)
#         peaks, _ = find_peaks(row_or_col, height=height, distance=len(row_or_col))
#         peaks_data.append(row_or_col[peaks][0])
#     return np.array(peaks_data)

def npeaks_feature(data, axis=0, height=None, distance=None):
    def peak_finder(arr):
        peak_indices, _ = find_peaks(arr, height=height, distance=distance)
        return peak_indices.size
    return np.apply_along_axis(peak_finder, axis, data)

# def npeaks_feature(data, axis=0, height=None, distance=None):
#     peaks_number = []
#     for i in range(data.shape[1-axis]):
#         row_or_col = data.take(i, axis=1-axis)
#         peaks, _ = find_peaks(row_or_col, height=height, distance=distance)
#         peaks_number.append(len(peaks))
#     return np.array(peaks_number)  

def muPeaks_feature(data, axis=0, height=None, distance=None):
    def peak_finder(arr):
        peak_indices, _ = find_peaks(arr, height=height, distance=distance)
        return np.mean(arr[peak_indices]) if peak_indices.size > 0 else 0
    return np.apply_along_axis(peak_finder, axis, data)

# def muPeaks_feature(data, axis=0, height=None, distance=None):
#     peaks_mean = []
#     for i in range(data.shape[1-axis]):
#         row_or_col = data.take(i, axis=1-axis)
#         peaks, _ = find_peaks(row_or_col, height=height, distance=distance)
#         peaks_mean.append(np.mean(row_or_col[peaks]) if len(peaks) > 0 else 0)
#     return np.array(peaks_mean)

def sumPeaks_feature(data, axis=0, height=None, distance=None):
    def peak_finder(arr):
        peak_indices, _ = find_peaks(arr, height=height, distance=distance)
        return np.sum(arr[peak_indices])
    return np.apply_along_axis(peak_finder, axis, data)

# def sumPeaks_feature(data, axis=0, height=None, distance=None):
#     peaks_sum = []
#     for i in range(data.shape[1-axis]):
#         row_or_col = data.take(i, axis=1-axis)
#         peaks, _ = find_peaks(row_or_col, height=height, distance=distance)
#         peaks_sum.append(np.sum(row_or_col[peaks]) if len(peaks) > 0 else 0)
#     return np.array(peaks_sum)

def stdPeaks_feature(data, axis=0, height=None, distance=None):
    def peak_finder(arr):
        peak_indices, _ = find_peaks(arr, height=height, distance=distance)
        return np.std(arr[peak_indices]) if peak_indices.size > 0 else 0
    return np.apply_along_axis(peak_finder, axis, data)

# def stdPeaks_feature(data, axis=0, height=None, distance=None):
#     peak_values = peaks_feature(data, axis=axis, height=height, distance=distance)
#     return np.array([np.std(values) if len(values) > 0 else 0 for values in peak_values])

def fft_feature(data, axis=None):
    fft_result = scipy_fft.fft(data, axis=axis)
    return np.abs(fft_result)  # Return the magnitude spectrum


if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility

    arrays = [
        np.random.randint(0, 10, size=100),  # 1D array of integers
        np.random.uniform(0.0, 10.0, size=100),  # 1D array of floats
        np.random.randint(0, 10, size=(10, 10)),  # 2D array of integers
        np.random.uniform(0.0, 10.0, size=(10, 10)),  # 2D array of floats
        np.random.randint(-10, 0, size=100),  # 1D array of signed integers
        np.random.randint(-10, 10, size=(10, 10))  # 2D array of signed integers
    ]

    for i, data in enumerate(arrays):
        kurtosis_result = kurtosis_feature(data, axis=1 if data.ndim == 2 else None)
        kurtosis_simplified_result = kurtosis_feature_simplified(data, axis=1 if data.ndim == 2 else None)
        kurtosis_similarity = np.allclose(kurtosis_result, kurtosis_simplified_result, rtol=1e-5, atol=1e-8)

        print(f"Array {i+1}:")
        print("Data:", data)
        print("Kurtosis (kurtosis_feature):", kurtosis_result)
        print("Kurtosis (kurtosis_feature_simplified):", kurtosis_simplified_result)
        print("Kurtosis Similarity:", kurtosis_similarity)
        print()
