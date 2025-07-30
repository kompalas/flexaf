import numpy as np
import os
import pandas as pd
import logging
from collections import OrderedDict
from src.selection import feature_costs_map
from keras.callbacks import EarlyStopping
from src.args import AccuracyMetric
from src.classifier import FCNNKerasWrapper
from src.dataset import get_dataset
from src.utils import transform_categorical
from src.features import create_features_data_from_array

logger = logging.getLogger(__name__)


def run_greedy_feature_selection(args):
    """Run greedy feature selection on the dataset."""
    hidden_nodes = [50]
    learning_rate = 0.001
    training_epochs = 100
    early_stop_patience = 5

    # Load data
    train_data, test_data, sampling_rates, dataset_sr = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=None,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification,
        test_size=args.test_size
    )
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
    y_train_categ = transform_categorical(y_train, num_classes)
    y_test_categ = transform_categorical(y_test, num_classes)

    num_features = x_train.shape[1]
    selected_features = []
    remaining_features = list(range(num_features))

    results = []

    # EarlyStopping callback to be passed to training
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=early_stop_patience,
                                        restore_best_weights=True)

    for _ in range(num_features):
        best_acc = -np.inf
        best_feat = None

        for feat in remaining_features:
            candidate_features = selected_features + [feat]
            logger.info(f"Evaluating feature set: {candidate_features}")

            x_train_sub = x_train[:, candidate_features]
            x_test_sub = x_test[:, candidate_features]
            feature_costs_sub = feature_costs[candidate_features]

            model = FCNNKerasWrapper(
                accuracy_metric=AccuracyMetric.Accuracy,
                tune=False,
                feature_costs=feature_costs_sub,
                num_features=len(candidate_features),
                num_classes=np.unique(y_train).shape[0],
                num_nodes=hidden_nodes,
                learning_rate=learning_rate,
                test_data=(x_test_sub, y_test)
            )
            acc = model.train(x_train_sub, y_train, 
                              x_test_sub, y_test, 
                              epochs=training_epochs,
                              callbacks=[early_stop_callback],
                              verbose=1)

            if acc > best_acc:
                best_acc = acc
                best_feat = feat

        if best_feat is None:
            break  # No improvement

        selected_features.append(best_feat)
        remaining_features.remove(best_feat)

        total_cost = np.sum(feature_costs[selected_features])

        result = {
            'num_features': len(selected_features),
            'selected_features': selected_features.copy(),
            'accuracy': best_acc,
            'cost': total_cost
        }
        results.append(result)
        logger.info(f"Selected features: {selected_features}, Accuracy: {best_acc:.4f}, Cost: {total_cost:.2f}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.resdir, 'greedy_selection_results.csv'), index=False)

    logger.info("Greedy feature selection completed and saved.")
