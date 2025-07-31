import logging
from collections import OrderedDict
import numpy as np
from src.dataset import get_dataset
from src.selection import feature_costs_map
from src.features import create_features_data_from_array
from src.classifier import set_extra_clf_params, get_classifier
from src.args import AccuracyMetric


logger = logging.getLogger(__name__)


def perform_basic_evaluation(args):
    """Perform a simple evaluation of the dataset with basic features and without tuning."""
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=None,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size
    )
    train_data, test_data = data
    x_train, y_train = train_data
    x_test, y_test = test_data
    num_sensors = x_train.shape[1]

    features_dict = OrderedDict([
        (sensor_id, ['min', 'max', 'sum', 'mean'])
        for sensor_id in range(0, num_sensors)
    ])
    feature_costs = np.array([
        feature_costs_map[feature] 
        for sensor_features in features_dict.values()
        for feature in sensor_features
    ])
    input_precisions = [args.default_inputs_precision] * num_sensors

    # NOTE: this forces no resampling during feature extraction (equal to the 'dataset_sampling_rate')
    #       change to 'sampling_rates' if you want to use different rates per sensor
    new_sampling_rates = [dataset_sr] * num_sensors

    x_train_features, y_train = create_features_data_from_array(
        data=x_train,
        labels=y_train,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        window_size=args.default_window_size,
        dataset_sampling_rate=dataset_sr,
        sampling_rates=new_sampling_rates,
        target_clock=args.performance_target
    )
    x_test_features, y_test = create_features_data_from_array(
        data=x_test,
        labels=y_test,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        window_size=args.default_window_size,
        dataset_sampling_rate=dataset_sr,
        sampling_rates=new_sampling_rates,
        target_clock=args.performance_target
    )

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
