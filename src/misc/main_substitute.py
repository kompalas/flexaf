import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import logging
from src.classifier import get_classifier, set_extra_clf_params, FCNNKerasWrapper
from src.custom_models.mlp.input_gate_layer import ConcreteGate
from src.args import AccuracyMetric
from src.utils import env_cfg
from src.dataset import get_dataset
from src.features import create_features_from_df_sliding


logger = logging.getLogger(__name__)


def rest():
    #tuner_gates.results_summary()
    model_gates = tuner_gates.get_best_models(num_models=1)[0]
    accuracy_gates = model_gates.evaluate(x_test, y_test_categ, verbose=0)[1]
    logger.info(f"Best tuned model with gates test accuracy: {accuracy_gates:.4f}")

    # get the best hyperparameters
    best_hp_gates = tuner_gates.get_best_hyperparameters()[0]
    for key, value in best_hp_gates.values.items():
        logger.info(f"Best {key.replace('_', ' ')}: {value}")

    neurons_used_gates = [best_hp_gates.get(f'neurons_l{i+1}') for i in range(best_hp_gates.values.get('num_layers'))]
    _info = {
        'method': 'with_gates_tuned',
        'neurons': '-'.join([str(i) for i in neurons_used_gates]),
        'accuracy': accuracy_gates,
    }
    info.append(_info)
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(args.resdir, 'results.csv'), index=False)
    logger.info(f"Best model with gates info: {_info}")


    # Prune features based on the gates
    logger.info("Pruning features based on the gates...")

    threshold = 0.1  # Pruning threshold

    # Extract the gate layer
    # gate_layer = model_gates.layers[0]
    # gates = gate_layer.gates.numpy()
    gate_layer = [l for l in model_gates.layers if isinstance(l, ConcreteGate)][0]
    gates = gate_layer.get_gate_values()
    logger.info(f"Feature gates: {gates}")

    # Select features with gates > threshold
    selected_indices = np.where(gates > threshold)[0]
    logger.info(f"Selected features: {selected_indices}")
    logger.info(f"Remaining {len(selected_indices)} / {len(gates)} features")

    # Prepare pruned data
    x_train_pruned = x_train[:, selected_indices]
    x_test_pruned = x_test[:, selected_indices]
    pruned_feature_costs = feature_costs[selected_indices]

    # Rebuild and retrain model
    extra_params['num_features'] = len(selected_indices)
    extra_params['feature_costs'] = pruned_feature_costs
    model_pruned = FCNNKerasWrapper(accuracy_metric=AccuracyMetric.Accuracy,
                                    tune=False,
                                    lambda_reg=None,
                                    num_nodes=neurons_used_gates,
                                    learning_rate=0.001,
                                    **extra_params)

    pruned_accuracy = model_pruned.train(x_train_pruned, y_train, x_test_pruned, y_test)
    logger.info(f"Classifier accuracy after pruning features: {pruned_accuracy:.4f}")

    _info = {
        'method': 'pruned_features',
        'neurons': '-'.join([str(i) for i in neurons_used_gates]),
        'accuracy': pruned_accuracy,
    }
    info.append(_info)
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(args.resdir, 'results.csv'), index=False)
    logger.info(f"Classifier accuracy after pruning features: {_info}")




def run():
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
    features_data, labels = create_features_from_df_sliding(data=data,
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
                                            input_precisions=input_precisions,
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



import os
import traceback
import logging
import numpy as np
import pandas as pd
import keras_tuner as kt
import keras
import tensorflow as tf
from functools import partial
from src.args import AccuracyMetric
from src.utils import env_cfg
from src.classifier import set_extra_clf_params, FCNNKerasWrapper
from src.dataset import get_dataset, split_data_with_costs


logger = logging.getLogger(__name__)



def main():

    args = env_cfg()
    args.resdir = logging.getLogger().logdir

    data, original_sampling_rates, dataset_sampling_rate = get_dataset(
        args.dataset_type, args.dataset_file,
        resampling_rate=None,
        binary_classification=args.binary_classification,
        three_class_classification=args.three_class_classification
    )
    train_data, test_data, categ_labels, feature_costs = split_data_with_costs(
        data, original_sampling_rate=dataset_sampling_rate
    )
    x_train, y_train = train_data
    x_test, y_test = test_data
    y_train_categ, y_test_categ = categ_labels

    extra_params = set_extra_clf_params(args.classifier_type,
                                        input_precisions=[args.default_inputs_precision] * x_train.shape[1],
                                        x_test=x_test,
                                        y_test=y_test,
                                        feature_costs=feature_costs)
    extra_params = {key: extra_params[key] for key in ['num_classes', 'num_features', 'num_samples', 'test_data']}


    hidden_layers = [1, 2, 3]
    hidden_neurons = [10, 20, 50, 100]
    search_trials = 50  # Number of trials for hyperparameter tuning
    training_epochs = 50  # Number of epochs for training
    num_nodes = [hidden_neurons[-1]]


    # First get the accuracy without the input gates
    classifier = FCNNKerasWrapper(accuracy_metric=AccuracyMetric.Accuracy,
                                  tune=False,
                                  feature_costs=None,
                                  num_nodes=num_nodes,
                                  learning_rate=0.001,
                                  **extra_params)

    # Train the model without input gates
    test_accuracy = classifier.train(x_train, y_train, x_test, y_test)
    logger.info(f"Classifier accuracy before tuning: {test_accuracy:.4f}")

    info = [{
        'method': 'no_gates',
        'model_id': id(classifier),
        'model_index': 0,
        'num_layers': len(num_nodes),
        'neurons': '-'.join([str(i) for i in num_nodes]),
        'gate_threshold': None,
        'accuracy': test_accuracy,
        'cost': np.sum(feature_costs),
        'num_features': extra_params['num_features'],
        'selected_features': list(range(extra_params['num_features']))
    }]
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(args.resdir, 'results.csv'), index=False)

    # Tune the model without input gates


    def builder_hp(hp, hidden_layers, hidden_neurons, input_gates, 
                   num_classes, input_bitwidth, num_features, feature_costs):
        """Builds a model for hyperparameter tuning."""
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')

        if input_gates:
            # Here, explore only the hyperparameters related to the input gates    
            lambda_reg = hp.Float('lambda_reg', 0.01, 10.0, sampling='log')
            temperature = hp.Float('temperature', 0.01, 1.0, sampling='log')
            # warmup_epochs = hp.Int('warmup_epochs', 1, 10, step=1)
            warmup_epochs = 5

            # The rest of the parameters are fixed
            num_layers = hidden_layers
            neurons_per_layer = hidden_neurons

        else:
            # Here, explore the hyperparameters related to the model architecture
            lambda_reg = temperature = warmup_epochs= None

            num_layers = hp.Choice('num_layers', hidden_layers)
            neurons_per_layer = []
            for i in range(num_layers):
                neurons = hp.Choice(f'neurons_l{i+1}', hidden_neurons)
                neurons_per_layer.append(neurons)

        return classifier.define_model(num_classes=num_classes,
                                       input_bitwidth=input_bitwidth,
                                       num_features=num_features,
                                       num_nodes=neurons_per_layer,
                                       learning_rate=learning_rate,
                                       feature_costs=feature_costs,
                                       lambda_reg=lambda_reg,
                                       temperature=temperature,
                                       warmup_epochs=warmup_epochs)
    
    # train the classifier without input gates
    exact_builder = partial(builder_hp,
                            hidden_layers=hidden_layers,
                            hidden_neurons=hidden_neurons,
                            input_gates=False,
                            num_classes=extra_params['num_classes'],
                            input_bitwidth=args.default_inputs_precision,
                            num_features=extra_params['num_features'],
                            feature_costs=feature_costs)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=5,
                                            restore_best_weights=True)
    callbacks = [early_stop]

    tuner = kt.BayesianOptimization(
        hypermodel=exact_builder,
        objective="val_accuracy",
        max_trials=search_trials,
        project_name=f"{args.resdir}/tuning_no_gates",
        overwrite=True,
    )
    tuner.search(x_train,
                 y_train_categ,
                 epochs=training_epochs,
                 validation_data=(x_test, y_test_categ),
                 verbose=1,
                 callbacks=callbacks)

    #tuna.results_summary()
    model = tuner.get_best_models(num_models=1)[0]
    accuracy = model.evaluate(x_test, y_test_categ, verbose=0)[1]
    logger.info(f"Best tuned model without gates test accuracy: {accuracy:.4f}")

    # get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    for key, value in best_hp.values.items():
        logger.info(f"Best {key.replace('_', ' ')}: {value}")

    neurons_used = [best_hp.get(f'neurons_l{i+1}') for i in range(best_hp.values.get('num_layers'))]

    _info = {
        'method': 'no_gates_tuned',
        'model_id': id(model),
        'model_index': 0,
        'num_layers': best_hp.values.get('num_layers'),
        'neurons': '-'.join([str(i) for i in neurons_used]),
        'gate_threshold': None,
        'accuracy': accuracy,
        'cost': np.sum(feature_costs),
        'num_features': extra_params['num_features'],
        'selected_features': list(range(extra_params['num_features']))
    }
    info.append(_info)
    df = pd.DataFrame(info)
    df.to_csv(os.path.join(args.resdir, 'results.csv'), index=False)
    logger.info(f"Best tuned model without gates info: {_info}")

    # Execute the same tuning for the model with input gates
    gate_builder = partial(builder_hp,
                           hidden_layers=len(neurons_used),
                           hidden_neurons=neurons_used,
                           input_gates=True,
                           num_classes=extra_params['num_classes'],
                           input_bitwidth=args.default_inputs_precision,
                           num_features=extra_params['num_features'],
                           feature_costs=feature_costs)

    # update callbacks to include epoch end for gates
    class GateEpochUpdater(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for layer in self.model.layers:
                if hasattr(layer, 'on_epoch_end'):
                    layer.on_epoch_end()
    callbacks = [GateEpochUpdater(), early_stop]

    tuner_gates = kt.BayesianOptimization(
        hypermodel=gate_builder,
        objective="val_accuracy",
        max_trials=search_trials,
        project_name=f"{args.resdir}/tuning_with_gates",
        overwrite=True,
    )

    tuner_gates.search(x_train,
                       y_train_categ,
                       epochs=training_epochs,
                       validation_data=(x_test, y_test_categ),
                       verbose=1,
                       callbacks=callbacks)

    all_models = tuner_gates.get_best_models(num_models=100)
    thresholds = [0.01, 0.05, 0.1, 0.2]  # pruning thresholds

    pareto_data = []
    for i, model in enumerate(all_models):
        # Extract gate values from ConcreteGate layer
        gate_layer = [l for l in model.layers if hasattr(l, 'get_gate_values')][0]
        gate_values = gate_layer.get_gate_values()
        logger.info(f"Model {i} gate values: {gate_values}")

        # Get the best hyperparameters for this model
        neurons_used_gates = [layer.units for layer in model.layers if 'hidden_dense' in layer.name]
        logger.info(f"Model {i} neurons used: {neurons_used_gates}")

        for thresh in thresholds:
            selected_idx = np.where(gate_values > thresh)[0]
            if len(selected_idx) == 0:
                continue  # Skip models that prune all features

            # Subset data and costs
            x_train_sub = x_train[:, selected_idx]
            x_test_sub = x_test[:, selected_idx]
            cost_sub = feature_costs[selected_idx]

            # Build and retrain pruned model (no gates now)
            model_pruned = FCNNKerasWrapper(
                accuracy_metric=AccuracyMetric.Accuracy,
                tune=False,
                feature_costs=cost_sub,
                lambda_reg=None,
                num_nodes=neurons_used_gates,  # from tuned config
                learning_rate=0.001,
                num_features=len(selected_idx),
                num_classes=extra_params['num_classes'],
                test_data=(x_test_sub, y_test)
            )
            acc = model_pruned.train(x_train_sub, y_train, x_test_sub, y_test)
            # Total feature cost = sum(gate * cost), or full cost if after pruning
            total_cost = np.sum(cost_sub)
            logger.info(f"Model {i} threshold {thresh} - accuracy: {acc:.4f}, cost: {total_cost:.2f}, features: {len(selected_idx)}")

            _info = {
                'method': 'gates',
                'model_id': id(model),
                'model_index': i,
                'num_layers': len(neurons_used_gates),
                'neurons': '-'.join([str(n) for n in neurons_used_gates]),
                'gate_threshold': thresh,
                'accuracy': acc,
                'cost': total_cost,
                'num_features': len(selected_idx),
                'selected_features': selected_idx.tolist()
            }
            info.append(_info)
            df = pd.DataFrame(info)
            df.to_csv(os.path.join(args.resdir, 'results.csv'), index=False)

