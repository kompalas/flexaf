import logging
import pandas as pd
import numpy as np
from functools import partial
from collections import OrderedDict
from src.args import DatasetType
from src.utils import normalization, resample_dataframe
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

__all__ = [
    'get_dataset', 'split_dataset',
    'get_wesad',
    'get_stress_in_nurses',
    'get_spd',
    'get_affective_road',
    'get_drive_db',
    'get_coswara',
    'get_har',
    'get_harth',
    'get_wisdm'
]


def get_dataset(dataset_type, dataset_file, resampling_rate=None, 
                binary_classification=False, three_class_classification=True,
                test_size=None):
    """Load the given dataset"""
    dataset_load_functions = {
        DatasetType.WESAD_RB: partial(get_wesad, 
                                      dataset_type=DatasetType.WESAD_RB,
                                      binary_classification=binary_classification,
                                      three_class_classification=three_class_classification),
        DatasetType.WESAD_E4: partial(get_wesad, 
                                      dataset_type=DatasetType.WESAD_E4,
                                      binary_classification=binary_classification,
                                      three_class_classification=three_class_classification),
        DatasetType.WESAD_Merged: partial(get_wesad, 
                                      dataset_type=DatasetType.WESAD_Merged,
                                      binary_classification=binary_classification,
                                      three_class_classification=three_class_classification),
        DatasetType.StressInNurses: get_stress_in_nurses,
        DatasetType.SPD: get_spd,
        DatasetType.AffectiveROAD: get_affective_road,
        DatasetType.DriveDB: get_drive_db,
        DatasetType.Coswara: get_coswara,
        DatasetType.HAR: get_har,
        DatasetType.HARTH: get_harth,
        DatasetType.WISDM: get_wisdm,
        DatasetType.DaphNET: get_daphnet
    }

    data = pd.read_csv(dataset_file, low_memory=False)
    try:
        data, sampling_rates, uniform_sampling_rate = dataset_load_functions[dataset_type](data)
    except KeyError:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    assert 'label' in data.columns, "A column 'label' must be present in the raw data"
    assert 'subject' in data.columns, "A column 'subject' must be present in the raw data"
    assert len(sampling_rates) > 0, "Sampling rates must be defined for the dataset"
    assert all(signal in data.columns for signal in sampling_rates.keys()), \
        "All signals must correspond to the defined sampling rates"
    assert all(max(rates) <= uniform_sampling_rate for rates in sampling_rates.values()), \
        "All sampling rates must be less than or equal to the uniform sampling rate (i.e., no upsampling)"

    # reorganize columns to have 'label' and 'subject' at the end
    columns = [col for col in data.columns if col not in ['label', 'subject']] + ['subject', 'label']
    data = data[columns]

    # normalize the data
    for column in data.columns[:-2]:  # Exclude 'label' and 'subject'
        data[column] = normalization(data[column])

    sampling_rates = OrderedDict([
        (signal, sampling_rates[signal]) for signal in data.columns 
        if signal not in ['label', 'subject']
    ]) 

    if resampling_rate is not None:
        data = resample_dataframe(data, uniform_sampling_rate, resampling_rate)
        uniform_sampling_rate = resampling_rate

    if test_size is None:
        return data, sampling_rates, uniform_sampling_rate

    train_data, test_data = split_dataset(data, test_size)
    return (train_data, test_data), sampling_rates, uniform_sampling_rate


def split_dataset(data, test_size=0.2):
    """Split the dataset into training and testing sets"""
    subjects = np.random.permutation(data['subject'].unique())
    num_train_subjects = int(len(subjects) * (1 - test_size))
    train_subjects = subjects[:num_train_subjects]
    test_subjects = subjects[num_train_subjects:]
    logger.info(f"Splitting dataset:\n\t{len(train_subjects)} training subjects: {train_subjects}\n\t{len(test_subjects)} testing subjects: {test_subjects}")

    train_data = data[data['subject'].isin(train_subjects)]
    test_data = data[data['subject'].isin(test_subjects)]

    return train_data, test_data
    # x_train = train_data.drop(columns=['subject', 'label']).values
    # y_train = train_data['label'].values
    # x_test = test_data.drop(columns=['subject', 'label']).values
    # y_test = test_data['label'].values

    # return (x_train, y_train), (x_test, y_test)


def get_wesad(data, dataset_type, binary_classification=False, three_class_classification=True):
    """Load the WESAD dataset and preprocess it for binary or three-class classification."""
    def reduce_wesad_classes(data):
        """Reduce the number of classes in the WESAD dataset
            In preprocessing (basic_seg): if labels[idx] not in [1, 2, 3]
            8 classes in total:
                0: baseline -> Keep for binary/three-class classification
                1: stress -> Keep for binary/three-class classification
                2: amusement -> Keep for three-class classification / Transform to 0 for binary classification
                3: meditation1
                4: rest
                5: meditation2
                6: ??
                7: ??
        """
        # TODO: Figure out which class to keep/transform for binary classification
        if not binary_classification and not three_class_classification:
            return data

        # in case of three-class classification, keep the first three classes
        selected_classes = sorted(data['label'].unique())[:3]
        data = data[data['label'].isin(selected_classes)]

        # in case of binary classification, transform amusement to baseline
        if binary_classification:
            data = data.replace({'label': 2}, {'label': 0}) # transform amusement to baseline
        return data

    data = reduce_wesad_classes(data)
    uniform_sampling_rate = 128
    sampling_rates = {'ax': [128, 64, 32, 16, 8, 4],
                        'ay': [128, 64, 32, 16, 8, 4],
                        'az': [128, 64, 32, 16, 8, 4],
                        'emg': [128, 64, 32, 16, 8, 4],
                        'eda': [128, 64, 32, 16, 8, 4],
                        'temp': [4],
                        'ecg': [128, 64, 32, 16, 8, 4],
                        'resp': [128, 64, 32, 16, 8, 4],
                        'ax_wrist': [32, 16, 8, 4],
                        'ay_wrist': [32, 16, 8, 4],
                        'az_wrist': [32, 16, 8, 4],
                        'eda_wrist': [4],
                        'temp_wrist': [4],
                        'bvp_wrist': [64, 32, 16, 8, 4]}

    if dataset_type == DatasetType.WESAD_RB:
        data = data.drop(columns=[column for column in data.columns if 'wrist' in column])

    elif dataset_type == DatasetType.WESAD_E4:
        data = data.drop(columns=[column for column in data.columns if 'wrist' not in column and 'label' not in column and 'subject' not in column])

    sampling_rates = {column: rates for column, rates in sampling_rates.items() 
                      if column in data.columns}
    return data, sampling_rates, uniform_sampling_rate


def get_stress_in_nurses(data):
    """Load the Stress in Nurses dataset and preprocess it."""
    data = data.rename(columns={'id': 'subject'}) 
    data = data.drop(columns=['datetime'])
    sampling_rates = {'X': [32, 16, 8, 4],
                    'Y': [32, 16, 8, 4],
                    'Z': [32, 16, 8, 4],
                    'EDA': [4],
                    'TEMP': [4],
                    'BVP': [64, 32, 16, 8, 4]}
    uniform_sampling_rate = 64
    return data, sampling_rates, uniform_sampling_rate


def get_spd(data):
    """Load the SPD dataset and preprocess it."""
    data = data.rename(columns={'Stress': 'label', 'ID': 'subject'})
    data = data.drop(columns=['Timestamp', 'HR', 'IBI_d'])
    sampling_rates = {'ACC_x': [32, 16, 8, 4],
                        'ACC_y': [32, 16, 8, 4],
                        'ACC_z': [32, 16, 8, 4],
                        'EDA': [4],
                        'TEMP': [4],
                        'BVP': [64, 32, 16, 8, 4]}
    uniform_sampling_rate = 64
    return data, sampling_rates, uniform_sampling_rate


def get_affective_road(data):
    """Load the AffectiveROAD dataset and preprocess it."""
    data = data.rename(columns={'STRESS': 'label', 'ID': 'subject'})
    sampling_rates = {'ACC_X': [32, 16, 8, 4],
                        'ACC_Y': [32, 16, 8, 4],
                        'ACC_Z': [32, 16, 8, 4],
                        'EDA': [4],
                        'TEMP': [4],
                        'BVP': [64, 32, 16, 8, 4]}
    uniform_sampling_rate = 64
    return data, sampling_rates, uniform_sampling_rate


def get_drive_db(data):
    """Load the DriveDB dataset and preprocess it."""
    data = data.rename(columns={'Stress': 'label'})
    sampling_rates = {'ECG': [16, 8, 4],
                        'EMG': [16, 8, 4],
                        'foot_EDA': [16, 8, 4],
                        'hand_EDA': [16, 8, 4],
                        'RESP': [16, 8, 4]}
    uniform_sampling_rate = 16
    return data, sampling_rates, uniform_sampling_rate


def get_coswara(data):
    """Load the Coswara dataset and preprocess it."""
    raise NotImplementedError("Coswara dataset loading is not implemented") 


def get_har(data):
    """Load the HAR dataset and preprocess it."""
    sampling_rates = {column: [50, 32, 16, 8, 4] for column in data.columns
                      if column not in ['label', 'subject']}
    uniform_sampling_rate = 50
    return data, sampling_rates, uniform_sampling_rate


def get_harth(data):
    """Load the HARTH dataset and preprocess it."""
    data = data.drop(columns=['timestamp'])
    sampling_rates = {column: [50, 32, 16, 8, 4] for column in data.columns
                      if column not in ['label', 'subject']}
    uniform_sampling_rate = 50
    return data, sampling_rates, uniform_sampling_rate


def get_wisdm(data):
    """Load the WISDM dataset and preprocess it."""
    sampling_rates = {column: [20, 16, 8, 4] for column in data.columns 
                      if column not in ['label', 'subject']}
    uniform_sampling_rate = 20
    return data, sampling_rates, uniform_sampling_rate


def get_daphnet(data):
    """Load the DaphNET dataset and preprocess it."""
    data = data.rename(columns={'individual': 'subject', 'annotation': 'label'})
    data = data.drop(columns=['time'])
    sampling_rates = {column: [64, 32, 16, 8, 4] for column in data.columns 
                      if column not in ['label', 'subject']}
    uniform_sampling_rate = 64
    return data, sampling_rates, uniform_sampling_rate