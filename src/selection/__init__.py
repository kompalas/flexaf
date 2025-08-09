import logging
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import GroupKFold
from src.classifier import set_extra_clf_params
from src.dataset import get_dataset
from src.utils import transform_categorical
from src.features import create_features_from_df_subjectwise


logger = logging.getLogger(__name__)

# Area, in mm^2
feature_costs_map = OrderedDict([
    ('min', 0.0020),
    ('max', 0.0020),
    # ('sum', 0.0265),
    ('sum', 0.0099 / 2),  # Adjusted to regularize the cost of sum and mean
    # ('mean', 0.0251)
    ('mean', 0.0084 / 2)  # Adjusted to regularize the cost of sum and mean,
])
feature_costs_map_placeholder = {'min': 4, 'max': 4, 'sum': 10, 'mean': 11}

kept_features = ['min', 'max', 'sum', 'mean']

all_features = ['min', 'max', 'mean', 'std', 'sum', 
                'skewness', 'kurtosis', 'range', 'median',
                'peaks', 'npeaks', 'muPeaks']


def prepare_feature_data(args, use_all_features=False):
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=args.uniform_resampling_rate,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size
    )
    train_data, test_data = data
    num_sensors = len(train_data.columns) - 2  # Exclude 'label' and 'subject'
    num_classes = len(train_data['label'].unique())

    features_dict = OrderedDict([
        (sensor_id, all_features) if use_all_features else
        (sensor_id, kept_features)
        for sensor_id in range(0, num_sensors)
    ])

    if use_all_features:
        feature_costs = None
    else:
        feature_costs = np.array([
            feature_costs_map[feature] 
            for sensor_features in features_dict.values()
            for feature in sensor_features
        ])

    input_precisions = [args.default_inputs_precision] * num_sensors
    new_sampling_rates = [dataset_sr] * num_sensors  # this forces no resampling during feature extraction (equal to the 'dataset_sampling_rate')
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
    y_train_categ = transform_categorical(y_train, num_classes)
    y_test_categ = transform_categorical(y_test, num_classes)

    extra_params = set_extra_clf_params(
        args.classifier_type,
        input_precisions=input_precisions,
        x_test=x_test, y_test=y_test,
        feature_costs=feature_costs
    )
    filtered_params = {k: extra_params[k] for k in ['num_classes', 'num_features', 'num_samples', 'test_data'] if k in extra_params}
    filtered_params['num_classes'] = num_classes
    return (x_train.values, y_train), (x_test.values, y_test), (y_train_categ, y_test_categ), feature_costs, filtered_params, input_precisions


def get_subject_cv_splits(data, n_splits=5):
    subject_ids = data['subject'].unique()
    gkf = GroupKFold(n_splits=n_splits)

    # Precompute subject-level splits
    cv_subject_folds = []
    for train_subjects_idx, val_subjects_idx in gkf.split(subject_ids, groups=subject_ids):
        train_subjects = subject_ids[train_subjects_idx]
        val_subjects = subject_ids[val_subjects_idx]
        cv_subject_folds.append((train_subjects, val_subjects))
    return cv_subject_folds


def prepare_feature_data_cross_validation(args, cv_folds=5):
    """Prepare feature data for cross-validation."""
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=args.uniform_resampling_rate,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=None  # No split, we will handle it in the cross-validation loop
    )
    num_sensors = len(data.columns) - 2  # Exclude 'label' and 'subject'
    num_classes = len(data['label'].unique())

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
    new_sampling_rates = [dataset_sr] * num_sensors  # this forces no resampling during feature extraction (equal to the 'dataset_sampling_rate')
    
    cv_subject_folds = get_subject_cv_splits(data, n_splits=cv_folds)
    return data, cv_subject_folds, feature_costs, features_dict, input_precisions, new_sampling_rates, dataset_sr, num_classes


def select_specific_features(data, dataset_type, subjects_to_keep, features_to_keep, save=True):
    """Select specific features from the dataset."""
    if isinstance(data, (list, tuple)) and len(data) == 2:
        # merge train and test data vertically
        data = data[0].append(data[1], ignore_index=True)

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
        logger.info(f"Sensor {sensor} ({pruned_data.columns[sensor]}): {feature_name}")
        if sensor not in features_dict_to_keep:
            features_dict_to_keep[sensor] = []
        features_dict_to_keep[sensor].append(feature_name)

    return pruned_data, features_dict_to_keep