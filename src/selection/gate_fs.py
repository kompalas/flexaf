import os
import logging
import numpy as np
import pandas as pd
import keras
import keras_tuner as kt
import tensorflow as tf
from functools import partial
from src.args import AccuracyMetric
from src.classifier import FCNNKerasWrapper
from src.selection import prepare_data

logger = logging.getLogger(__name__)


def train_baseline_model(x_train, y_train, x_test, y_test, num_nodes, learning_rate, **extra_params):
    classifier = FCNNKerasWrapper(
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=False,
        num_nodes=num_nodes,
        learning_rate=learning_rate,
        **extra_params
    )
    acc = classifier.train(x_train, y_train, x_test, y_test)
    return acc, classifier


def builder_hp(hp, hidden_layers, hidden_neurons, input_gates,
               num_classes, input_bitwidth, num_features, feature_costs):
    from src.classifier import FCNNKerasWrapper  # avoid circular import

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')
    
    if input_gates:
        lambda_reg = hp.Float('lambda_reg', 0.01, 10.0, sampling='log')
        temperature = hp.Float('temperature', 0.01, 1.0, sampling='log')
        warmup_epochs = 5
        neurons_per_layer = hidden_neurons
    else:
        lambda_reg = temperature = warmup_epochs = None
        num_layers = hp.Choice('num_layers', hidden_layers)
        neurons_per_layer = [hp.Choice(f'neurons_l{i+1}', hidden_neurons) for i in range(num_layers)]

    return FCNNKerasWrapper.define_model(
        num_classes=num_classes,
        input_bitwidth=input_bitwidth,
        num_features=num_features,
        num_nodes=neurons_per_layer,
        learning_rate=learning_rate,
        feature_costs=feature_costs,
        lambda_reg=lambda_reg,
        temperature=temperature,
        warmup_epochs=warmup_epochs
    )


def run_tuning(builder, x_train, y_train, x_test, y_test, training_epochs, max_trials, callbacks, project_name):
    tuner = kt.BayesianOptimization(
        hypermodel=builder,
        objective="val_accuracy",
        max_trials=max_trials,
        project_name=project_name,
        overwrite=True,
    )
    tuner.search(x_train, y_train,
                 epochs=training_epochs,
                 validation_data=(x_test, y_test),
                 verbose=1,
                 callbacks=callbacks)
    return tuner


def prune_and_evaluate(models, thresholds, x_train, x_test, y_train, y_test, feature_costs, extra_params):
    results = []
    for i, model in enumerate(models):
        gate_layer = [l for l in model.layers if hasattr(l, 'get_gate_values')][0]
        gate_values = gate_layer.get_gate_values()
        
        neurons_used = [l.units for l in model.layers if 'hidden_dense' in l.name]

        for thresh in thresholds:
            selected = np.where(gate_values > thresh)[0]
            if len(selected) == 0:
                continue
            x_train_sub = x_train[:, selected]
            x_test_sub = x_test[:, selected]
            cost_sub = feature_costs[selected]

            clf = FCNNKerasWrapper(
                accuracy_metric=AccuracyMetric.Accuracy,
                tune=False,
                feature_costs=cost_sub,
                lambda_reg=None,
                num_nodes=neurons_used,
                learning_rate=0.001,
                num_features=len(selected),
                num_classes=extra_params['num_classes'],
                test_data=(x_test_sub, y_test)
            )
            acc = clf.train(x_train_sub, y_train, x_test_sub, y_test)
            results.append({
                'method': 'gates',
                'model_id': id(model),
                'model_index': i,
                'gate_threshold': thresh,
                'accuracy': acc,
                'cost': np.sum(cost_sub),
                'num_features': len(selected),
                'selected_features': selected.tolist(),
                'neurons': '-'.join(map(str, neurons_used))
            })
    return results


def log_and_save_results(info, resdir):
    df = pd.DataFrame(info)
    out_path = os.path.join(resdir, 'results.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")


def run_differentiable_feature_selection(args):
    """Run the differentiable feature selection experiment with ConcreteGate."""
    train_data, test_data, categ_labels, feature_costs, extra_params = prepare_data(args)
    x_train, y_train = train_data
    x_test, y_test = test_data
    y_train_categ, y_test_categ = categ_labels

    # -------------------- CONTROLLED HYPERPARAMETERS --------------------
    hidden_layers = [1, 2, 3]
    hidden_neurons = [10, 20, 50, 100]
    training_epochs = 2
    search_trials = 3
    initial_learning_rate = 0.001
    num_nodes = [hidden_neurons[-1]]
    thresholds = [0.01, 0.05, 0.1, 0.2]
    # -------------------------------------------------------------------

    acc, clf = train_baseline_model(x_train, y_train, x_test, y_test, num_nodes, initial_learning_rate, **extra_params)
    base_result = [{
        'method': 'no_gates',
        'model_id': id(clf),
        'model_index': 0,
        'gate_threshold': None,
        'accuracy': acc,
        'cost': np.sum(feature_costs),
        'num_features': extra_params['num_features'],
        'selected_features': list(range(extra_params['num_features'])),
        'neurons': '-'.join(map(str, num_nodes))
    }]
    log_and_save_results(base_result, args.resdir)
    # -------------------------------------------------------------------

    builder_no_gates = partial(builder_hp, 
                               hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons, 
                               input_gates=False,
                               num_classes=extra_params['num_classes'],
                               input_bitwidth=args.default_inputs_precision,
                               num_features=extra_params['num_features'],
                               feature_costs=feature_costs)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner = run_tuning(builder_no_gates, x_train, y_train_categ, x_test, y_test_categ,
                       training_epochs, search_trials, [early_stop],
                       project_name=f"{args.resdir}/tuning_no_gates")

    best_model = tuner.get_best_models(num_models=1)[0]
    acc = best_model.evaluate(x_test, y_test_categ, verbose=0)[1]
    hp = tuner.get_best_hyperparameters()[0]
    neurons = [hp.get(f'neurons_l{i+1}') for i in range(hp.get('num_layers'))]

    tuned_result = [{
        'method': 'no_gates_tuned',
        'model_id': id(best_model),
        'model_index': 0,
        'gate_threshold': None,
        'accuracy': acc,
        'cost': np.sum(feature_costs),
        'num_features': extra_params['num_features'],
        'selected_features': list(range(extra_params['num_features'])),
        'neurons': '-'.join(map(str, neurons))
    }]
    log_and_save_results(base_result + tuned_result, args.resdir)
    # -------------------------------------------------------------------

    builder_with_gates = partial(builder_hp, hidden_layers=len(neurons),
                                 hidden_neurons=neurons, input_gates=True,
                                 num_classes=extra_params['num_classes'],
                                 input_bitwidth=args.default_inputs_precision,
                                 num_features=extra_params['num_features'],
                                 feature_costs=feature_costs)

    class GateEpochUpdater(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for layer in self.model.layers:
                if hasattr(layer, 'on_epoch_end'):
                    layer.on_epoch_end()

    tuner_gates = run_tuning(builder_with_gates, x_train, y_train_categ, x_test, y_test_categ,
                             training_epochs, search_trials, [GateEpochUpdater(), early_stop],
                             project_name=f"{args.resdir}/tuning_with_gates")

    gate_models = tuner_gates.get_best_models(num_models=100)
    pruned_info = prune_and_evaluate(gate_models, thresholds, x_train, x_test,
                                     y_train, y_test, feature_costs, extra_params)

    log_and_save_results(base_result + tuned_result + pruned_info, args.resdir)
