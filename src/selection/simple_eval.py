import logging
from collections import OrderedDict
import numpy as np
from src.dataset import get_dataset
from src.selection import feature_costs_map
from src.features import create_features_from_df_subjectwise
from src.classifier import set_extra_clf_params, get_classifier
from src.args import AccuracyMetric


logger = logging.getLogger(__name__)


def select_specific_features(data, dataset_type, selected_features):
    # Change these to get the correct features
    subjects_to_keep = ['S15', 'S4', 'S9']
    features_to_keep = [1, 16, 17, 20, 25, 32, 41, 45]

    # Filter the data to keep only the specified subjects and save to csv
    pruned_data = data[data['subject'].isin(subjects_to_keep)]
    pruned_data.to_csv(f'{dataset_type.name.lower()}_32hz_pruned.csv', index=False)

    # Create a features dictionary based on the specified features to keep
    num_sensors = len(pruned_data.columns) - 2
    features_dict = OrderedDict([
        (sensor_id, selected_features)
        for sensor_id in range(0, num_sensors)
    ])
    all_feature_names = [
        f"{sensor_id}_{feature}"
        for sensor_id, features in features_dict.items()
        for feature in features
    ]
    features_names_to_keep = np.array(all_feature_names)[features_to_keep]
    for feature in features_names_to_keep:
        sensor, feature_name = feature.split('_')
        logger.info(f"Sensor {sensor} ({pruned_data.columns[sensor]}): {feature_name}")


def perform_basic_evaluation(args):
    """Perform a simple evaluation of the dataset with basic features
    """
    selected_features = ['mean', 'max', 'min', 'sum']

    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=args.uniform_resampling_rate,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size,
        # test_size=None,
    )
    # select_specific_features(data, args.dataset_type, selected_features)
    # data.to_csv(f'{args.dataset_type.name.lower()}_32hz.csv', index=False)

    train_data, test_data = data
    num_sensors = len(train_data.columns) - 2  # Exclude 'label' and 'subject'

    features_dict = OrderedDict([
        (sensor_id, selected_features)
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
