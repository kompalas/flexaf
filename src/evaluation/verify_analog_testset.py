import tensorflow as tf
import os
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import transform_categorical, convert_to_fixed_point
from src.args import DatasetType
from src.dataset import get_dataset
from src.selection.simple_eval import select_specific_features
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
    accuracy = model.evaluate(x_test_sub, y_test_categ, verbose=1)[1]
    print(f"Pruned accuracy: {accuracy:.5f}")
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
    remade_accuracy = model.evaluate(x_test_remade, y_test_remade_categ, verbose=1)[1]
    print(f"Remade test set accuracy: {remade_accuracy:.5f}")
    return x_test_remade, y_test_remade_categ, remade_accuracy


def merge_analog_features(*feature_files, remove_x_columns=False, input_precision=None, offset=0.0):
    """Merge multiple feature files into a single dataframe."""
    merged_features = []
    for feature_file in feature_files:
        assert os.path.exists(feature_file) and feature_file.endswith('.csv'), f"Invalid feature file: {feature_file}"
        features = pd.read_csv(feature_file)
        if remove_x_columns:
            columns_to_remove = [col for col in features.columns if col.endswith(' X')]
            features = features.drop(columns=columns_to_remove)
        merged_features.append(features)

    # concatenate all feature dataframes horizontally
    merged_df = pd.concat(merged_features, axis=1)
    # subtract offset from all values
    merged_df = merged_df - offset

    if input_precision is None:
        return merged_df.values

    # convert each feature column to fixed-point representation
    features_data = []
    for feature_id in range(merged_df.shape[1]):
        fxp_data = convert_to_fixed_point(
            merged_df.values[:, feature_id], 
            precision=input_precision, fractional_bits=input_precision,
            normalize=None, rescale=True, signed=False, 
        )
        features_data.append(fxp_data)
    return np.array(features_data).T


def mse(a, b): return np.mean((a - b) ** 2)
def mae(a, b): return np.mean(np.abs(a - b))


if __name__ == "__main__":

    # load the saved model and test set and evaluate its accuracy
    expdir = '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.06-16.16.48.524'
    fold = 0
    sparsity = 0.5
    trial = 2
    model, x_test_sub, y_test_categ, accuracy = load_and_evaluate_model(expdir, fold, sparsity, trial)

    # verify the accuracy with newly extracted features
    dataset_type = DatasetType.WESAD_Merged
    dataset_file = 'data/wesad_merged.csv'
    resampling_rate = 32
    subjects_to_keep = ['S15', 'S4', 'S9']
    features_to_keep = [1, 16, 17, 20, 25, 32, 41, 45]
    x_test_remade, y_test_remade_categ, remade_accuracy = verify_feature_extraction_and_evaluation(
        model, dataset_type, dataset_file, resampling_rate, subjects_to_keep, features_to_keep,
        input_precision=32
    )

    # load the analog test set features
    feature_file1 = '/home/balaskas/flexaf/data/analog/wesad_tuned_max_renamed.csv'
    feature_file2 = '/home/balaskas/flexaf/data/analog/wesad_tuned_mean_renamed.csv'
    x_test_analog = merge_analog_features(
        feature_file1, feature_file2, 
        remove_x_columns=True, input_precision=None, offset=1.0
    )

    # add labels from originally pruned test set
    original_df = pd.read_csv('/home/balaskas/flexaf/data/wesad_tuned.csv')
    labels = original_df['label'].values  # assume labels are resampled at the given rate, so no need to resample them
    # create non-overlapping windows of labels and keep the most common one in each window
    reshaped_labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=resampling_rate)[::resampling_rate]
    filtered_labels = stats.mode(reshaped_labels, axis=1)[0]
    y_test_analog = transform_categorical(filtered_labels, num_classes=model.output_shape[-1])

    # evaluate the model on the analog test set
    analog_accuracy = model.evaluate(x_test_analog, y_test_analog, verbose=1)[1]
    print(f"Analog test set accuracy: {analog_accuracy:.5f}")

    # calculate the mse between each column of the original test set and the analog test set
    mse_values = [mse(x_test_sub[:, i], x_test_analog[:, i]) for i in range(x_test_sub.shape[1])]
    mae_values = [mae(x_test_sub[:, i], x_test_analog[:, i]) for i in range(x_test_sub.shape[1])]
    print("MSE between original and analog test set features:", mse_values)
    print("MAE between original and analog test set features:", mae_values)
    print("MSE mean:", np.mean(mse_values))
    print("MAE mean:", np.mean(mae_values))