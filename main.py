import os
import traceback
import logging
import pickle
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from src.args import AccuracyMetric
from src.utils import env_cfg
from src.dataset import get_dataset
from src.features import create_features_data
from src.classifier import get_classifier, set_extra_clf_params


logger = logging.getLogger(__name__)


def main():
    args = env_cfg()
    args.resdir = logging.getLogger().logdir
    with open(os.path.join(args.resdir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    new_sampling_rate = 32

    data, original_sampling_rates, dataset_sampling_rate = get_dataset(args.dataset_type, args.dataset_file,
                                                                       resampling_rate=new_sampling_rate,  # NOTE: This is hardcoded for now
                                                                       binary_classification=args.binary_classification,
                                                                       three_class_classification=args.three_class_classification)
    num_sensors = len(data.columns) - 1

    features_dict = OrderedDict([
        (sensor_id, {'min', 'max', 'sum', 'mean'})
        for sensor_id, sensor_name in enumerate(data.columns) if sensor_name != 'label'
    ])
    feature_costs_map = {'min': 4, 'max': 4, 'sum': 10, 'mean': 11}
    feature_costs = np.array([feature_costs_map[feature] for sensor_features in features_dict.values() for feature in sensor_features])

    input_precisions = [args.default_inputs_precision] * num_sensors
    new_sampling_rates = [new_sampling_rate] * num_sensors
    features_data, labels = create_features_data(data=data,
                                                features_dict=features_dict, 
                                                inputs_precisions=input_precisions, 
                                                window_size=args.default_window_size,
                                                dataset_sampling_rate=dataset_sampling_rate,
                                                sampling_rates=new_sampling_rates,
                                                target_clock=args.performance_target)

    x_selected = features_data.values
    features_data['label'] = labels
    # TODO: Split this so that time-series data is not split randomly
    x_train, x_test, y_train, y_test = train_test_split(x_selected, labels,
                                                        test_size=args.test_size,
                                                        stratify=labels,
                                                        random_state=args.global_seed)
    
    extra_clf_params = set_extra_clf_params(args.classifier_type,
                                            adc_precisions=input_precisions,
                                            x_test=x_test,
                                            y_test=y_test,
                                            feature_costs=feature_costs)
    classifier = get_classifier(args.classifier_type,
                                accuracy_metric=AccuracyMetric.Accuracy,
                                tune=args.tune_classifier,
                                train_data=(x_train, y_train),
                                seed=args.global_seed,
                                **extra_clf_params)

    test_accuracy = classifier.train(x_train, y_train, x_test, y_test)
    logger.info(f"Classifier accuracy before pruning features: {test_accuracy:.4f}")

    threshold = 0.05  # Pruning threshold

    # Extract the gate layer
    if not hasattr(classifier, '_clf') or not hasattr(classifier._clf, 'layers'):
        return
        
    gate_layer = classifier._clf.layers[0]
    gates = gate_layer.gates.numpy()

    # Select features with gates > threshold
    selected_indices = np.where(gates > threshold)[0]
    logger.info(f"Selected features: {selected_indices}")
    logger.info(f"Remaining {len(selected_indices)} / {len(gates)} features")

    # Prepare pruned data
    x_train_pruned = x_train[:, selected_indices]
    x_test_pruned = x_test[:, selected_indices]
    pruned_feature_costs = feature_costs[selected_indices]

    # Rebuild and retrain model
    extra_clf_params['num_features'] = len(selected_indices)
    extra_clf_params['feature_costs'] = pruned_feature_costs
    model_pruned = get_classifier(args.classifier_type,
                                    accuracy_metric=AccuracyMetric.Accuracy,
                                    tune=args.tune_classifier,
                                    train_data=(x_train_pruned, y_train),
                                    seed=args.global_seed,
                                    **extra_clf_params)

    pruned_accuracy = model_pruned.train(x_train_pruned, y_train, x_test_pruned, y_test)
    logger.info(f"Classifier accuracy after pruning features: {pruned_accuracy:.4f}")



if __name__ == '__main__':
    try:
        main()
    except Exception:
        if logger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = logger.handlers
            logger.handlers = [h for h in logger.handlers if type(h) != logging.StreamHandler]
            logger.error(traceback.format_exc())
            logger.handlers = handlers_bak
        raise
    except KeyboardInterrupt:
        logger.info("")
        logger.info("--- Keyboard Interrupt ---")
    finally:
        if logger.handlers:
            logfiles = [handler.baseFilename for handler in logger.handlers if
                        type(handler) == logging.FileHandler]
            logger.info(f"Log file(s) for this run in {' | '.join(logfiles)}")