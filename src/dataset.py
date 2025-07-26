import logging
import pandas as pd
from collections import OrderedDict
from src.args import DatasetType
from src.utils import normalization, resample_dataframe


logger = logging.getLogger(__name__)


def get_dataset(dataset_type, dataset_file, resampling_rate=None, binary_classification=False, three_class_classification=True):
    """Load the raw data from the dataset file and reduce the number of classes if necessary"""
    raw_data = pd.read_csv(dataset_file, low_memory=False)
    sampling_rates = {}
    uniform_sampling_rate = 64

    if dataset_type in (DatasetType.WESAD_RB, DatasetType.WESAD_E4, DatasetType.WESAD_Merged):
        raw_data = raw_data.select_dtypes(include=['number'])
        raw_data = reduce_wesad_classes(raw_data, binary_classification, three_class_classification)
        if dataset_type == DatasetType.WESAD_RB:
            raw_data = raw_data.drop(columns=[column for column in raw_data.columns if 'wrist' in column])
            uniform_sampling_rate = 128
            sampling_rates = {'ax': [128, 64, 32, 16, 8, 4],
                              'ay': [128, 64, 32, 16, 8, 4],
                              'az': [128, 64, 32, 16, 8, 4],
                              'emg': [128, 64, 32, 16, 8, 4],
                              'eda': [128, 64, 32, 16, 8, 4],
                              'temp': [4],
                              'ecg': [128, 64, 32, 16, 8, 4],
                              'resp': [128, 64, 32, 16, 8, 4]}

        elif dataset_type == DatasetType.WESAD_E4:
            raw_data = raw_data.drop(columns=[column for column in raw_data.columns if 'wrist' not in column and 'label' not in column])
            uniform_sampling_rate = 128
            sampling_rates = {'ax_wrist': [32, 16, 8, 4],
                              'ay_wrist': [32, 16, 8, 4],
                              'az_wrist': [32, 16, 8, 4],
                              'eda_wrist': [4],
                              'temp_wrist': [4],
                              'bvp_wrist': [64, 32, 16, 8, 4]}

        elif dataset_type == DatasetType.WESAD_Merged:
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

    elif dataset_type == DatasetType.StressInNurses:
        raw_data = raw_data.select_dtypes(include=['number'])
        for column in raw_data.columns:
            if column != 'label':
                raw_data[column] = normalization(raw_data[column])
        sampling_rates = {'X': [32, 16, 8, 4],
                          'Y': [32, 16, 8, 4],
                          'Z': [32, 16, 8, 4],
                          'EDA': [4],
                          'TEMP': [4],
                          'BVP': [64, 32, 16, 8, 4]}

    elif dataset_type == DatasetType.SPD:
        raw_data = raw_data.rename(columns={'Stress': 'label'})
        raw_data = raw_data.drop(columns=['Timestamp', 'HR', 'ID', 'IBI_d'])
        for column in raw_data.columns:
            if column != 'label':
                raw_data[column] = normalization(raw_data[column])
        sampling_rates = {'ACC_x': [32, 16, 8, 4],
                          'ACC_y': [32, 16, 8, 4],
                          'ACC_z': [32, 16, 8, 4],
                          'EDA': [4],
                          'TEMP': [4],
                          'BVP': [64, 32, 16, 8, 4]}

    elif dataset_type == DatasetType.AffectiveROAD:
        raw_data = raw_data.rename(columns={'STRESS': 'label'})
        for column in raw_data.columns:
            if column != 'label':
                raw_data[column] = normalization(raw_data[column])
        sampling_rates = {'ACC_X': [32, 16, 8, 4],
                          'ACC_Y': [32, 16, 8, 4],
                          'ACC_Z': [32, 16, 8, 4],
                          'EDA': [4],
                          'TEMP': [4],
                          'BVP': [64, 32, 16, 8, 4]}

    elif dataset_type == DatasetType.DriveDB:
        raw_data = raw_data.rename(columns={'Stress': 'label'})
        for column in raw_data.columns:
            if column != 'label':
                raw_data[column] = normalization(raw_data[column])
        sampling_rates = {'ECG': [16, 8, 4],
                          'EMG': [16, 8, 4],
                          'foot_EDA': [16, 8, 4],
                          'hand_EDA': [16, 8, 4],
                          'RESP': [16, 8, 4]}
        uniform_sampling_rate = 16

    elif dataset_type == DatasetType.Coswara:
        raise NotImplementedError("Coswara dataset is not yet supported.")
    
    elif dataset_type == DatasetType.HAR:
        raise NotImplementedError("HAR dataset is not yet supported.")
    
    elif dataset_type == DatasetType.HARTH:
        raise NotImplementedError("HARTH dataset is not yet supported.")
    
    elif dataset_type == DatasetType.WISDM:
        raise NotImplementedError("WISDM dataset is not yet supported.")

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    assert 'label' in raw_data.columns, "A column 'label' must be present in the raw data"
    assert len(sampling_rates) > 0, "Sampling rates must be defined for the dataset"
    assert all(signal in raw_data.columns for signal in sampling_rates.keys()), \
        "All signals must correspond to the defined sampling rates"
    assert all(max(rates) <= uniform_sampling_rate for rates in sampling_rates.values()), \
        "All sampling rates must be less than or equal to the uniform sampling rate (i.e., no upsampling)"

    sampling_rates = OrderedDict([(signal, sampling_rates[signal]) for signal in raw_data.columns if signal != 'label']) 

    if resampling_rate is not None:
        resampled_data = resample_dataframe(raw_data, uniform_sampling_rate, resampling_rate)
        return resampled_data, sampling_rates, uniform_sampling_rate
    return raw_data, sampling_rates, uniform_sampling_rate


def reduce_wesad_classes(data, binary_classification=False, three_class_classification=False):
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