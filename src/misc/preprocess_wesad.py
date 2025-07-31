import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from scipy import stats
from pyEDA.main import resample_data, normalization
import argparse


class one_subject_prepare:
    def __init__(self, subject, path):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        self.chest_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        #assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        return wrist_data

    def get_chest_data(self):
        #assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data


def segment_labels_kb(labels, target_size):
    # segment labels into target_size
    window_size = len(labels) // target_size
    trunc_labels = labels[:target_size*window_size]
    windows = trunc_labels.reshape(target_size, window_size)
    return stats.mode(windows, axis=1)[0].flatten()


def raw_data_extract(path, resampling_rate=100):
    pop_feature = pd.DataFrame()

    for subject_path in path.iterdir():
        if not subject_path.is_dir():
            continue

        subject = one_subject_prepare(subject_path.name, path)
        com_feature = pd.DataFrame()

        # get data from chest and wrist sensors
        chest_data_dict = subject.get_chest_data()
        wrist_data_dict = subject.get_wrist_data()
        labels = subject.get_labels()

        # get all features
        ax = np.array(chest_data_dict['ACC'])[:,0]
        ay = np.array(chest_data_dict['ACC'])[:,1]
        az = np.array(chest_data_dict['ACC'])[:,2]
        emg = np.array(chest_data_dict['EMG'].flatten())
        temp = np.array(chest_data_dict['Temp'].flatten())
        eda = np.array(chest_data_dict['EDA'].flatten())
        ecg = np.array(chest_data_dict['ECG'].flatten())
        resp = np.array(chest_data_dict['Resp'].flatten())
        ax_wrist = np.array(wrist_data_dict['ACC'])[:,0]
        ay_wrist = np.array(wrist_data_dict['ACC'])[:,1]
        az_wrist = np.array(wrist_data_dict['ACC'])[:,2]
        eda_wrist = np.array(wrist_data_dict['EDA'].flatten())
        temp_wrist = np.array(wrist_data_dict['TEMP'].flatten())
        bvp_wrist = np.array(wrist_data_dict['BVP'].flatten())  # keep only bvp from wrist sensor

        # resample each other signal without extracting the features
        str_lst = ['ax', 'ay', 'az', 'emg', 'temp', 'eda', 'ecg', 'resp']
        sig_lst = [ax, ay, az, emg, temp, eda, ecg, resp]
        for _, (name, sig) in enumerate(zip(str_lst, sig_lst)):

            # only resample and normalize the signal
            resampled_sig = resample_data(sig, 700, resampling_rate)
            com_feature[name] = normalization(resampled_sig)

        sig_length_to_reach = len(resampled_sig)

        str_lst = ['ax_wrist', 'ay_wrist', 'az_wrist', 'eda_wrist', 'temp_wrist', 'bvp_wrist']
        sig_lst = [ax_wrist, ay_wrist, az_wrist, eda_wrist, temp_wrist, bvp_wrist]
        original_rates = [32, 32, 32, 4, 4, 64]
        for _, (name, sig, original_rate) in enumerate(zip(str_lst, sig_lst, original_rates)):

            if resampling_rate == original_rate:
                resampled_sig = sig
            else:
                new_sample_rate = sig_length_to_reach * 64 / len(sig)
                resampled_sig = resample_data(sig, 64, new_sample_rate)
            com_feature[name] = normalization(resampled_sig)

        resampled_labels = segment_labels_kb(labels, len(resampled_sig))
        assert len(resampled_labels) == len(resampled_sig), f"{len(labels)}, {len(resampled_labels)}, {len(resampled_sig)}"
        com_feature['label'] = resampled_labels
        com_feature['subject'] = subject_path.name

        print(f"Subject {subject_path.name} data extracted")
        print(com_feature)
        print()

        # concatenate the created features
        pop_feature = pd.concat([pop_feature, com_feature], axis=0, ignore_index=True)

    return pop_feature


if __name__ == "__main__":
    # WESAD dataset paper: https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf

    # parser = argparse.ArgumentParser(description='Extract features from raw data')
    # parser.add_argument('--save-data-file', type=str, default=None, help='Output file to save the extracted data')
    # parser.add_argument('--resampling-rate', type=int, default=100, help='Resampling rate for the signals')
    # args = parser.parse_args()



    path = Path(__file__).parent.parent.parent / 'data' / 'wesad_raw' / 'WESAD'
    features_df = raw_data_extract(path, resampling_rate=128)
    features_df.to_csv('wesad_extracted.csv', index=False)

    # save the extracted features
    # if args.save_data_file is None:
    #     args.save_data_file = path.parent / f'extracted_r{args.resampling_rate}.csv'
    # features_df.to_csv(args.save_data_file, index=False)