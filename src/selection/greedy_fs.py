import numpy as np
import os
import pandas as pd
import logging
from collections import OrderedDict
from src.selection import feature_costs_map
from keras.callbacks import EarlyStopping
from src.args import AccuracyMetric
from src.classifier import FCNNKerasWrapper, ClassifierType, get_classifier, set_extra_clf_params
from src.dataset import get_dataset
from src.utils import transform_categorical
from src.features import create_features_data_from_array

logger = logging.getLogger(__name__)


def train_classifier(clf_type, x_train, y_train, x_test, y_test, feature_costs, input_precisions,
                     tune=True, seed=None, training_epochs=100):
    """Train the classifier with the given parameters."""
    extra_params = set_extra_clf_params(
        clf_type,
        input_precisions=input_precisions,
        x_test=x_test, y_test=y_test,
        feature_costs=feature_costs
    )
    if 'num_classes' in extra_params:
        extra_params['num_classes'] = len(np.unique(y_train))

    classifier = get_classifier(clf_type, accuracy_metric=AccuracyMetric.Accuracy,
                                tune=tune, train_data=(x_train, y_train),
                                seed=seed, **extra_params
    )
    # Train the model
    train_params = {}
    if clf_type in (ClassifierType.BNN, ClassifierType.TNN, ClassifierType.FCNN):
        train_params = {'epochs': training_epochs, 'verbose': 1, 
                        'callbacks': [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]}
        
    accuracy = classifier.train(x_train, y_train, x_test, y_test, **train_params)
    return accuracy


def run_greedy_feature_selection(args):
    """Run greedy feature selection on the dataset."""
    training_epochs = 100

    # Load data
    data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=None,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size
    )
    train_data, test_data = data
    x_train_raw, y_train = train_data
    x_test_raw, y_test = test_data
    num_classes = len(np.unique(y_train))
    num_sensors = x_train_raw.shape[1]

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

    x_train, y_train = create_features_data_from_array(
        data=x_train_raw,
        labels=y_train,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        window_size=args.default_window_size,
        dataset_sampling_rate=dataset_sr,
        sampling_rates=new_sampling_rates,
        target_clock=args.performance_target
    )
    x_test, y_test = create_features_data_from_array(
        data=x_test_raw,
        labels=y_test,
        features_dict=features_dict,
        inputs_precisions=input_precisions,
        window_size=args.default_window_size,
        dataset_sampling_rate=dataset_sr,
        sampling_rates=new_sampling_rates,
        target_clock=args.performance_target
    )

    results = []

    # first execute a baseline run with all features
    baseline_acc = train_classifier(
        args.classifier_type, x_train, y_train, x_test, y_test,
        feature_costs, input_precisions,
        tune=args.tune_classifier, seed=args.global_seed,
        training_epochs=training_epochs
    )
    logger.info(f"Baseline accuracy with all features: {baseline_acc:.4f}")
    results.append({
        'num_features': x_train.shape[1],
        'selected_features': list(range(x_train.shape[1])),
        'accuracy': baseline_acc,
        'cost': np.sum(feature_costs)
    })

    num_features = x_train.shape[1]
    selected_features = []
    remaining_features = list(range(num_features))

    for _ in range(num_features):
        best_acc = -np.inf
        best_feat = None

        for feat in remaining_features:
            candidate_features = selected_features + [feat]
            logger.info(f"Evaluating feature set: {candidate_features}")

            x_train_sub = x_train[:, candidate_features]
            x_test_sub = x_test[:, candidate_features]
            feature_costs_sub = feature_costs[candidate_features]

            acc = train_classifier(
                args.classifier_type, x_train_sub, y_train, x_test_sub, y_test,
                feature_costs_sub, input_precisions,
                tune=args.tune_classifier, seed=args.global_seed,
                training_epochs=training_epochs
            )

            if acc > best_acc:
                best_acc = acc
                best_feat = feat

        if best_feat is None:
            break  # No improvement

        selected_features.append(best_feat)
        remaining_features.remove(best_feat)

        total_cost = np.sum(feature_costs[selected_features])

        results.append({
            'num_features': len(selected_features),
            'selected_features': selected_features.copy(),
            'accuracy': best_acc,
            'cost': total_cost
        })
        logger.info(f"Selected features: {selected_features}, Accuracy: {best_acc:.4f}, Cost: {total_cost:.2f}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.resdir, 'greedy_selection_results.csv'), index=False)

    logger.info("Greedy feature selection completed and saved.")
