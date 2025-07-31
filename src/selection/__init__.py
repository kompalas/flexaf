from src.classifier import set_extra_clf_params
from src.dataset import get_dataset
from src.utils import transform_categorical
from src.features import create_features_data_from_array
import logging
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)


feature_costs_map = {'min': 4, 'max': 4, 'sum': 10, 'mean': 11}


def prepare_data(args):
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
    num_classes = len(np.unique(y_train))
    num_sensors = x_train.shape[1]

    features_dict = OrderedDict([
        (sensor_id, {'min', 'max', 'sum', 'mean'})
        for sensor_id in range(0, num_sensors)
    ])
    feature_costs = np.array([
        feature_costs_map[feature] 
        for sensor_features in features_dict.values()
        for feature in sensor_features
    ])
    
    input_precisions = [args.default_inputs_precision] * num_sensors
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
    y_train_categ = transform_categorical(y_train, num_classes)
    y_test_categ = transform_categorical(y_test, num_classes)

    extra_params = set_extra_clf_params(
        args.classifier_type,
        input_precisions=input_precisions,
        x_test=x_test_features,
        y_test=y_test,
        feature_costs=feature_costs
    )
    filtered_params = {k: extra_params[k] for k in ['num_classes', 'num_features', 'num_samples', 'test_data']}
    return (x_train_features, y_train), (x_test_features, y_test), (y_train_categ, y_test_categ), feature_costs, filtered_params, input_precisions
