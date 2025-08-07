import logging
import numpy as np
import pandas as pd
from collections import OrderedDict
from src.dataset import get_dataset
from src.selection import feature_costs_map, kept_features
from src.features import create_features_from_array_sliding
from src.features import create_features_from_df_subjectwise
from src.classifier import set_extra_clf_params, get_classifier
from src.args import AccuracyMetric


logger = logging.getLogger(__name__)


def perform_basic_evaluation_random_split(args):
    """Perform a simple evaluation of the dataset with basic features
    """
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=args.uniform_resampling_rate,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=None#args.test_size
    )

    num_sensors = len(data.columns) - 2
    # Uncomment the following lines if you want to use a specific train/test split
    # train_data, test_data = data
    # x_train, y_train = train_data
    # x_test, y_test = test_data
    # num_sensors = x_train.shape[1]

    features_dict = OrderedDict([
        (sensor_id, kept_features)
        for sensor_id in range(0, num_sensors)
    ])
    feature_costs = np.array([
        feature_costs_map[feature] 
        for sensor_features in features_dict.values()
        for feature in sensor_features
    ])
    input_precisions = [args.default_inputs_precision] * num_sensors

    # NOTE: this forces no resampling during feature extraction (equal to the 'dataset_sampling_rate')
    new_sampling_rates = [dataset_sr] * num_sensors
    # change to this if you want to use different rates per sensor
    # new_sampling_rates = [max(rates) for sensor, rates in sampling_rates.items()]


    # x_train_features, y_train = create_features_data_from_array(
    #     data=x_train,
    #     labels=y_train,
    #     features_dict=features_dict,
    #     inputs_precisions=input_precisions,
    #     window_size=args.default_window_size,
    #     dataset_sampling_rate=dataset_sr,
    #     sampling_rates=new_sampling_rates,
    #     target_clock=args.performance_target
    # )
    # x_test_features, y_test = create_features_data_from_array(
    #     data=x_test,
    #     labels=y_test,
    #     features_dict=features_dict,
    #     inputs_precisions=input_precisions,
    #     window_size=args.default_window_size,
    #     dataset_sampling_rate=dataset_sr,
    #     sampling_rates=new_sampling_rates,
    #     target_clock=args.performance_target
    # )

    from sklearn.model_selection import train_test_split
    from src.features import create_features_from_df_sliding
    features_data, labels = create_features_from_df_sliding(
        data=data,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        window_size=args.default_window_size,
        sampling_rates=new_sampling_rates,
        target_clock=args.performance_target
    )
    x_selected = features_data.values
    features_data['label'] = labels
    x_train_features, x_test_features, y_train, y_test = train_test_split(x_selected, labels,
                                                        test_size=args.test_size,
                                                        random_state=args.global_seed)



    extra_params = set_extra_clf_params(
        args.classifier_type,
        input_precisions=input_precisions,
        x_test=x_test_features,
        y_test=y_test,
        feature_costs=feature_costs
    )
    classifier = get_classifier(
        args.classifier_type,
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=args.tune_classifier,
        train_data=(x_train_features, y_train),
        seed=args.global_seed,
        **extra_params
    )
    accuracy = classifier.train(x_train_features, y_train,
                                x_test_features, y_test)
    logger.info(f"Accuracy: {accuracy:.4f}")


def select_specific_features(data, dataset_type, subjects_to_keep, features_to_keep, save=True):
    """Select specific features from the dataset."""
    if isinstance(data, (list, tuple)) and len(data) == 2:
        # merge train and test dataframes vertically
        data = pd.concat(data, ignore_index=True)

    # Filter the data to keep only the specified subjects and save to csv
    pruned_data = data[data['subject'].isin(subjects_to_keep)]
    if save:
        pruned_data.to_csv(f'{dataset_type.name.lower()}_32hz_pruned.csv', index=False)

    # Create a features dictionary based on the specified features to keep
    num_sensors = len(pruned_data.columns) - 2
    features_dict = OrderedDict([
        (sensor_id, kept_features)
        for sensor_id in range(0, num_sensors)
    ])
    all_feature_names = [
        f"{sensor_id}_{feature}"
        for sensor_id, features in features_dict.items()
        for feature in features
    ]
    features_names_to_keep = np.array(all_feature_names)[features_to_keep]
    features_dict_to_keep = OrderedDict()
    for feature in features_names_to_keep:
        sensor, feature_name = feature.split('_')
        sensor = int(sensor)
        print(f"Sensor {sensor} ({pruned_data.columns[sensor]}): {feature_name}")
        if sensor not in features_dict_to_keep:
            features_dict_to_keep[sensor] = []
        features_dict_to_keep[sensor].append(feature_name)

    return pruned_data, features_dict_to_keep


def perform_basic_evaluation_single_fold(args):
    """Perform a simple evaluation of the dataset with basic features
    """
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=args.uniform_resampling_rate,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size,
        # test_size=None,
    )

    # # Change these to get the correct features
    # subjects_to_keep = ['S15', 'S4', 'S9']
    # features_to_keep = [1, 16, 17, 20, 25, 32, 41, 45]
    # select_specific_features(data, args.dataset_type, selected_features, subjects_to_keep, features_to_keep)
    # data.to_csv(f'{args.dataset_type.name.lower()}_32hz.csv', index=False)

    train_data, test_data = data
    num_sensors = len(train_data.columns) - 2  # Exclude 'label' and 'subject'

    features_dict = OrderedDict([
        (sensor_id, kept_features)
        for sensor_id in range(0, num_sensors)
    ])
    feature_costs = np.array([
        feature_costs_map[feature] 
        for sensor_features in features_dict.values()
        for feature in sensor_features
    ])
    # input_precisions = [args.default_inputs_precision] * num_sensors
    input_precisions = [16] * num_sensors  # uncomment for close-to floating-point accuracy

    # NOTE: this forces no resampling during feature extraction (equal to the 'dataset_sampling_rate')
    new_sampling_rates = [dataset_sr] * num_sensors
    # change to this if you want to use different rates per sensor
    # new_sampling_rates = [max(rates) for sensor, rates in sampling_rates.items()]

    x_train, y_train = create_features_from_df_subjectwise(
        data=train_data,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        sampling_rates=new_sampling_rates,
        original_sampling_rate=dataset_sr,
        window_size=args.default_window_size,
        target_clock=args.performance_target
    )
    x_test, y_test = create_features_from_df_subjectwise(
        data=test_data,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        sampling_rates=new_sampling_rates,
        original_sampling_rate=dataset_sr,
        window_size=args.default_window_size,
        target_clock=args.performance_target
    )

    extra_params = set_extra_clf_params(
        args.classifier_type,
        input_precisions=input_precisions,
        x_test=x_test, y_test=y_test,
        feature_costs=feature_costs
    )
    classifier = get_classifier(
        args.classifier_type,
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=args.tune_classifier,
        train_data=(x_train, y_train),
        seed=args.global_seed,
        **extra_params
    )
    accuracy = classifier.train(x_train, y_train, x_test, y_test)
    logger.info(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":

    from src.args import DatasetType

    dataset_type = DatasetType.HARTH
    dataset_file = 'data/harth.csv'
    subjects_to_keep = ['S012', 'S017', 'S022', 'S027', 'S029']
    features_to_keep = [0, 1, 5, 12, 16, 17, 20]

    data, sampling_rates, dataset_sr = get_dataset(
        dataset_type, dataset_file,
        resampling_rate=32,
        binary_classification=False,
        three_class_classification=True,
        test_size=None,
        # test_size=None,
    )
    select_specific_features(data, dataset_type, subjects_to_keep, features_to_keep, save=True)