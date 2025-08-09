import tensorflow as tf
import os
import numpy as np
import pandas as pd
import re
from scipy import stats
from src.utils import transform_categorical, convert_to_fixed_point
from src.args import DatasetType
from src.dataset import get_dataset
from src.selection.simple_eval_bkp import select_specific_features
from src.features import create_features_from_df_subjectwise
from src.selection import kept_features


def load_and_evaluate_model(expdir, fold, sparsity, trial):
    """Load a pruned model and evaluate its accuracy on the test set."""
    model_name = f'gate_model_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.keras'
    model = tf.keras.models.load_model(
        os.path.join(expdir, 'results', 'classifiers', model_name),
    )

    x_test_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_test_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_test = np.load(os.path.join(expdir, 'results', 'data', f'y_test_fold{fold}.npy'))
    y_test_categ = transform_categorical(y_test, num_classes=model.output_shape[-1])

    # get the accuracy of the model on the pruned test set
    accuracy = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
    print(f"Original accuracy: {accuracy:.5f}")
    return model, x_test_sub, y_test_categ, accuracy


def verify_feature_extraction_and_evaluation(model, dataset_type, dataset_file, 
                                             resampling_rate, subjects_to_keep,features_to_keep, 
                                             input_precision=4, window_size=1, target_clock=1):
    data, sampling_rates, dataset_sr = get_dataset(
        dataset_type, dataset_file,
        resampling_rate=resampling_rate,
        binary_classification=False,
        three_class_classification=True,
        test_size=None,
    )
    pruned_data, features_dict = select_specific_features(data, dataset_type, subjects_to_keep, features_to_keep, save=False)
    x_test_remade, y_test_remade = create_features_from_df_subjectwise(
        data=pruned_data,
        features_dict=features_dict,
        inputs_precisions=[input_precision] * pruned_data.shape[1],
        sampling_rates=[dataset_sr] * pruned_data.shape[1],
        original_sampling_rate=dataset_sr,
        window_size=window_size,
        target_clock=target_clock
    )
    y_test_remade_categ = transform_categorical(y_test_remade, num_classes=model.output_shape[-1])
    # evaluate the model on the remade test set
    remade_accuracy = model.evaluate(x_test_remade, y_test_remade_categ, verbose=0)[1]
    print(f"Remade test set accuracy: {remade_accuracy:.5f}")
    return x_test_remade, y_test_remade_categ, remade_accuracy, data.columns


def post_process_features(features, offset=1.0, N=3, vref=1.5, inputshift=1.0):
    """Post-process features by removing ' X' suffix from column names."""
    for column in features.columns:
        if column.endswith(' X'):
            continue  # skip columns with ' X' suffix

        if 'min' in column.lower() or 'max' in column.lower() or 'mean' in column.lower():
            features[column] = features[column] - offset
        elif 'sum' in column.lower():
            features[column] = features[column] + ((N - 1) * vref) - (N * inputshift)
    return features


def merge_analog_features(*feature_files, remove_x_columns=False, input_precision=None, 
                          offset=1.0, N=3, vref=1.5, inputshift=1.0):
    """Merge multiple feature files into a single dataframe."""
    merged_features = []
    for feature_file in feature_files:
        if feature_file is None:
            continue
        assert os.path.exists(feature_file) and feature_file.endswith('.csv'), f"Invalid feature file: {feature_file}"
        features = pd.read_csv(feature_file)
        features = features.drop(columns=['Unnamed: 0', 'Unnamed: 1'], errors='ignore')
        if remove_x_columns:
            columns_to_remove = [col for col in features.columns if col.endswith(' X')]
            features = features.drop(columns=columns_to_remove)

        # NOTE: Do not uncomment, this is specific for one use case (harth)
        # features['VMin_sensor6 Y'], features['VMax_sensor6 Y'] = features['VMax_sensor6 Y'].copy(), features['VMin_sensor6 Y'].copy()
        # switch min and max columns
        # for sensor in [1, 2, 3, 4, 5, 6]:
        #     vmax_col = f"VMax_sensor{sensor} Y"
        #     vmin_col = f"VMin_sensor{sensor} Y"
        #     features[vmax_col], features[vmin_col] = features[vmin_col].copy(), features[vmax_col].copy()

        processed_features = post_process_features(features, offset, N, vref, inputshift)
        merged_features.append(processed_features)

    # concatenate all feature dataframes horizontally
    merged_df = pd.concat(merged_features, axis=1)
    # subtract offset from all values
    # merged_df = merged_df - offset

    if input_precision is None:
        return merged_df

    # convert each feature column to fixed-point representation
    features_data = []
    for feature_id in range(merged_df.shape[1]):
        fxp_data = convert_to_fixed_point(
            merged_df.values[:, feature_id], 
            precision=input_precision, fractional_bits=input_precision,
            normalize=None, rescale=True, signed=False, 
        )
        features_data.append(fxp_data)
    return pd.DataFrame(np.array(features_data).T, columns=merged_df.columns)


def mse(a, b): return np.mean((a - b) ** 2)
def mae(a, b): return np.mean(np.abs(a - b))


def align_dataframe_columns(df1, df2, all_sensor_names):
    # if more than the needed features are present, keep only the ones that were used in the model
    if df1.shape[1] != df2.shape[1]:
        assert df2.shape[1] > df1.shape[1], "In case of column mismatch, the analog df should have more columns than the remade df."
        print(f"WARNING: Analog test set has {analog_df.shape[1]} features, but the model expects {x_test_remade_np.shape[1]} features. Trimming...")

    sensor_ids = {sensor: i+1 for i, sensor in enumerate(all_sensor_names) if sensor not in ['label', 'subject']}
    df1_sensors_features = [('_'.join(col.split('_')[:-1]), col.split('_')[-1])  for col in df1.columns] 
    columns_df2_to_keep = []
    for sensor, feature in df1_sensors_features:
        if feature == 'mean' and f'VMean_sensor{sensor_ids[sensor]} Y' not in df2.columns:
            column_df2_name = f"Vmean_sensor{sensor_ids[sensor]} Y"
        elif feature == 'sum' and f'VSum_sensor{sensor_ids[sensor]} Y' not in df2.columns:
            column_df2_name = f"Vsum_sensor{sensor_ids[sensor]} Y"
        else:
            column_df2_name = f"V{feature.capitalize()}_sensor{sensor_ids[sensor]} Y"
        columns_df2_to_keep.append(column_df2_name)
    df2 = df2[columns_df2_to_keep]

    assert df1.shape[1] == df2.shape[1], "After trimming, the number of features should match."
    return df2

    # Extract sensor names and feature names from df1
    df1_colnames = df1.columns.tolist()
    # TODO: This assumes sensor names are in the format 'sensorX_featureY' where sensorX might include underscores, but features do not
    df1_sensors = ['_'.join(col.split('_')[:-1]) for col in df1_colnames] 
    df1_features = [col.split('_')[-1] for col in df1_colnames]

    # Create the mapping from sorted sensor numbers in df2 to sensor names in df1
    df2_sensor_info = []
    for col in df2.columns:
        match = re.match(r'V(\w+)_sensor(\d+)\s+Y', col)
        if not match:
            raise ValueError(f"Invalid column name format: {col}")
        feature, sensor_number = match.groups()
        df2_sensor_info.append((col, feature.lower(), int(sensor_number)))

    # Sort sensor numbers and map to df1 sensor names
    sorted_sensor_numbers = sorted(sensor_number for _, _, sensor_number in df2_sensor_info)
    if len(set(sorted_sensor_numbers)) != len(set(df1_sensors)):
        raise ValueError("Mismatch in number of unique sensors")
    sensor_mapping = dict(zip(sorted_sensor_numbers, df1_sensors))

    # Build mapping from df1 column names to df2 column names
    col_mapping = {}
    for df1_sensor, df1_feature in zip(df1_sensors, df1_features):
        for col, df2_feature, sensor_number in df2_sensor_info:
            if sensor_mapping[sensor_number] == df1_sensor and df2_feature == df1_feature:
                col_mapping[f"{df1_sensor}_{df1_feature}"] = col
                break
        else:
            raise ValueError(f"No match found in df2 for {df1_sensor}_{df1_feature}")

    # Reorder df2
    df2_reordered = df2[[col_mapping[col] for col in df1.columns]]
    return df2_reordered


if __name__ == "__main__":

    # expdir = '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.08-08.00.07.792'
    # fold = 4
    # sparsity = 0.2
    # trial = 1
    # dataset_type = DatasetType.WESAD_Merged
    # dataset_file = 'data/wesad_merged.csv'
    # subjects_to_keep = ['S10', 'S16', 'S5']
    # features_to_keep = [20, 45]
    # # feature_file1 = '/home/balaskas/flexaf/data/analog/Wesad_pruned_minmax_renamed__diff_fs_fcnn_wesad_merged___2025.08.08-08.00.07.792.csv'
    # feature_file1 = '/home/balaskas/flexaf/data/analog/Wesad_pruned_onlymax_renamed__diff_fs_fcnn_wesad_merged___2025.08.08-08.00.07.792.csv'
    # feature_file2 = '/home/balaskas/flexaf/data/analog/Wesad_pruned_onlymin_renamed__diff_fs_fcnn_wesad_merged___2025.08.08-08.00.07.792.csv'

    # expdir = '/home/balaskas/flexaf/saved_logs/stressinnurses/diff_fs_fcnn_stressinnurses___2025.08.08-11.59.24.548'
    # fold = 1
    # sparsity = 0.05
    # trial = 1
    # dataset_type = DatasetType.StressInNurses
    # dataset_file = 'data/stress_in_nurses.csv'
    # subjects_to_keep =  ['7E', '15', 'CE']
    # features_to_keep = [1, 5]
    # feature_file1 = '/home/balaskas/flexaf/data/analog/StressInNurses_Max_renamed__diff_fs_fcnn_stressinnurses___2025.08.08-11.59.24.548.csv'
    # feature_file2 = None

    # expdir = '/home/balaskas/flexaf/saved_logs/harth/diff_fs_fcnn_harth___2025.08.07-19.46.46.517'
    # fold, trial, sparsity = 0, 4, 0.1  # acc: 0.896663, only min max
    # # fold, trial, sparsity = 0, 1, 0.5  # acc: 0.892909, only min max again
    # # fold, trial, sparsity = 0, 3, 0.1  # acc: 0.888774, includes mean/max/min
    # dataset_type = DatasetType.HARTH
    # dataset_file = 'data/harth.csv'
    # subjects_to_keep = ['S012', 'S017', 'S022', 'S027', 'S029']  # fold 0
    # features_to_keep = [0, 1, 5, 12, 16, 17, 20]
    # # features_to_keep = [0, 1, 5, 9, 12, 20]
    # # features_to_keep = [0, 1, 8, 9, 11, 13, 15, 16, 17, 20, 21]
    # feature_file1 = '/home/balaskas/flexaf/data/analog/Harth_pruned_minmax_renamed__diff_fs_fcnn_harth___2025.08.07-19.46.46.517.csv'
    # feature_file2 = '/home/balaskas/flexaf/data/analog/Harth_pruned_mean_renamed__diff_fs_fcnn_harth___2025.08.07-19.46.46.517.csv'  # only for mean features

    expdir = '/home/balaskas/flexaf/saved_logs/spd/diff_fs_fcnn_spd___2025.08.06-20.55.49.900'
    fold, trial, sparsity = 1, 4, 0.2
    # fold, trial, sparsity = 1, 1, 0.5
    dataset_type = DatasetType.SPD
    dataset_file = 'data/spd.csv'
    subjects_to_keep = [2, 7, 12, 17, 22, 27, 32]
    features_to_keep = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23]
    # features_to_keep = [1, 2, 3, 4, 5, 6, 8, 9, 13, 17, 20, 21]
    feature_file1 = '/home/balaskas/flexaf/data/analog/spd_pruned_mean_sum_renamed__diff_fs_fcnn_spd___2025.08.06-20.55.49.900.csv'
    feature_file2 = '/home/balaskas/flexaf/data/analog/spd_pruned_min_max_renamed__diff_fs_fcnn_spd___2025.08.06-20.55.49.900.csv'

    # load the saved model and test set and evaluate its accuracy
    model, x_test_sub, y_test_categ, accuracy = load_and_evaluate_model(expdir, fold, sparsity, trial)

    # verify the accuracy with newly extracted features
    resampling_rate = 32
    x_test_remade, y_test_remade_categ, remade_accuracy, sensor_names = verify_feature_extraction_and_evaluation(
        model, dataset_type, dataset_file, resampling_rate, subjects_to_keep, features_to_keep,
        input_precision=31
    )
    x_test_remade_np = x_test_remade.values

    # load the analog test set features
    analog_df = merge_analog_features(
        feature_file1, feature_file2, #feature_file3,
        remove_x_columns=True, input_precision=32,
    )

    # align the number of samples in the analog test set with the remade test set
    if x_test_remade.shape[0] != analog_df.shape[0]:
        min_length = min(x_test_remade.shape[0], analog_df.shape[0])
        analog_df = analog_df[:min_length]
        x_test_remade = x_test_remade[:min_length]
        x_test_remade_np = x_test_remade_np[:min_length, :]
        y_test_remade_categ = y_test_remade_categ[:min_length, :]

    analog_df = align_dataframe_columns(x_test_remade, analog_df, sensor_names)
    x_test_analog = analog_df.values

    # add labels from originally pruned test set
    # original_df = pd.read_csv('/home/balaskas/flexaf/data/analog/wesad_merged_32hz_pruned.csv')
    # labels = original_df['label'].values  # assume labels are resampled at the given rate, so no need to resample them
    # # create non-overlapping windows of labels and keep the most common one in each window
    # reshaped_labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=resampling_rate)[::resampling_rate]
    # filtered_labels = stats.mode(reshaped_labels, axis=1)[0]
    # y_test_analog = transform_categorical(filtered_labels, num_classes=model.output_shape[-1])

    # evaluate the model on the analog test set
    analog_accuracy = model.evaluate(x_test_analog, y_test_remade_categ, verbose=0)[1]
    print(f"Analog test set accuracy: {analog_accuracy:.5f}")

    # calculate the mse between each column of the original test set and the analog test set
    mse_values = [mse(x_test_remade_np[:, i], x_test_analog[:, i]) for i in range(x_test_remade_np.shape[1])]
    mae_values = [mae(x_test_remade_np[:, i], x_test_analog[:, i]) for i in range(x_test_remade_np.shape[1])]
    print("MSE between original and analog test set features:", mse_values)
    print("MAE between original and analog test set features:", mae_values)
    print("MSE mean:", np.mean(mse_values))
    print("MAE mean:", np.mean(mae_values))

    # get the per-feature errors
    for feature_name in ['min', 'max', 'mean', 'sum']:
        column_ids_with_feature = [i for i, col in enumerate(x_test_remade.columns) if feature_name in col]
        if len(column_ids_with_feature) == 0:
            continue
        feature_mse = np.mean([mse_values[i] for i in column_ids_with_feature])
        feature_mae = np.mean([mae_values[i] for i in column_ids_with_feature])
        print(f"{feature_name} feature MSE: {feature_mse:.5f}, MAE: {feature_mae:.5f}")
    print()