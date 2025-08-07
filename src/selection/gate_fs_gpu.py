import numpy as np
import keras_tuner as kt
import tensorflow as tf
import logging
import os
import pandas as pd
from keras import layers, models
from keras.callbacks import EarlyStopping, Callback
from src.selection import prepare_feature_data_cross_validation
from src.features import create_features_from_df_subjectwise
from src.utils import transform_categorical
from src.classifier import FCNNKerasWrapper
from src.args import AccuracyMetric
from src.custom_models.mlp.input_gate_layer_gpu import ConcreteGate


logger = logging.getLogger(__name__)

# tf.debugging.set_log_device_placement(True)


def run_gated_model_pruning_experiment(args):
    """Run the gated model pruning experiment."""
    search_trials = 30
    training_epochs = 50
    batch_size = 1024
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    num_models_to_keep = min(5, search_trials)

    # confirm that TensorFlow runs on a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 0 will be used
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs available:\n{gpus}")

    # create directory for saving results
    results_dir = os.path.join(args.resdir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_data_dir = os.path.join(results_dir, 'data')
    os.makedirs(results_data_dir, exist_ok=True)
    results_clf_dir = os.path.join(results_dir, 'classifiers')
    os.makedirs(results_clf_dir, exist_ok=True)

    # prepare data for cross-validation
    cv_folds = int(1/args.test_size)
    data, cv_subject_folds, feature_costs, features_dict, input_precisions, sampling_rates, dataset_sr, num_classes = \
        prepare_feature_data_cross_validation(args, cv_folds)
    
    results = []
    for fold, (train_subjects, test_subjects) in enumerate(cv_subject_folds):
        train_data = data[data['subject'].isin(train_subjects)]
        test_data = data[data['subject'].isin(test_subjects)]
        logger.info(f"Running fold {fold + 1}/{cv_folds} with subjects: "
                    f"train={train_subjects.tolist()}, test={test_subjects.tolist()}")

        x_train, y_train = create_features_from_df_subjectwise(
            data=train_data,
            features_dict=features_dict,
            inputs_precisions=[16] * len(input_precisions),  # use close-to floating-point precision for training
            sampling_rates=sampling_rates,
            original_sampling_rate=dataset_sr,
            window_size=args.default_window_size,
            target_clock=args.performance_target
        )
        x_test, y_test = create_features_from_df_subjectwise(
            data=test_data,
            features_dict=features_dict,
            inputs_precisions=input_precisions,
            sampling_rates=sampling_rates,
            original_sampling_rate=dataset_sr,
            window_size=args.default_window_size,
            target_clock=args.performance_target
        )
        y_train_categ = transform_categorical(y_train, num_classes)
        y_test_categ = transform_categorical(y_test, num_classes)
        num_features = x_train.shape[1]
        x_train = x_train.values.astype(np.float32)
        x_test = x_test.values.astype(np.float32)

        # save the training and test data
        np.save(os.path.join(results_data_dir, f"x_train_fold{fold}.npy"), x_train)
        np.save(os.path.join(results_data_dir, f"y_train_fold{fold}.npy"), y_train)
        np.save(os.path.join(results_data_dir, f"x_test_fold{fold}.npy"), x_test)
        np.save(os.path.join(results_data_dir, f"y_test_fold{fold}.npy"), y_test)

        # from src.selection.statistical import fisher_select
        # features_to_use = fisher_select(x_train, y_train, k=num_features//2)  # select half of the features
        # fisher_score_mask = np.zeros(num_features, dtype=np.float32)
        # fisher_score_mask[features_to_use] = 1.0

        # === STEP 1: Warmup training without the gate ===
        def build_warmup_model():
            inputs = tf.keras.Input(shape=(num_features,), name='input')

            x = inputs
            for i, units in enumerate([100]):  # same structure as gated model
                x = tf.keras.layers.Dense(units, activation='relu', name=f"hidden_dense_{i}")(x)

            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            return model

        # Train warmup model for a few epochs
        warmup_epochs = 5
        warmup_model = build_warmup_model()
        warmup_model.fit(x_train, y_train_categ,
                        epochs=warmup_epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test_categ),
                        verbose=1)

        # === STEP 2: Define a gated model with hyperparameters ===
        def build_gated_model(hp):
            learning_rate = hp.Float('learning_rate', 1e-3, 5e-2, sampling='log')
            lambda_reg = hp.Float('lambda_reg', 1e-9, 1e-4, sampling='log')
            temperature = hp.Float('temperature', 1, 100, sampling='log')

            mask = None
            neurons = [100]

            inputs = tf.keras.Input(shape=(num_features,), name='input')

            gates = ConcreteGate(
                num_features=num_features,
                feature_costs=feature_costs,
                lambda_reg=lambda_reg,
                temperature=temperature,
                # warmup_epochs=warmup_epochs,  # no warmup now
                initial_binary_mask=mask,
                name='input_gate_layer'
            )(inputs)

            x = gates
            for i, units in enumerate(neurons):
                x = tf.keras.layers.Dense(units, activation='relu', name=f"hidden_dense_{i}")(x)

            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            # Transfer weights from warmup model
            for layer in model.layers:
                try:
                    pretrained_layer = warmup_model.get_layer(name=layer.name)
                    layer.set_weights(pretrained_layer.get_weights())
                except ValueError:
                    # Layer does not exist in warmup model (like gate) — skip
                    pass

            return model

        # === STEP 3: Use Keras Tuner with gated model ===
        tuner = kt.BayesianOptimization(
            hypermodel=build_gated_model,
            objective="val_accuracy",
            max_trials=search_trials,
            project_name=f"{args.resdir}/tuning_with_gates",
            overwrite=True
        )

        tuner.search(x_train, y_train_categ,
                    epochs=training_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test_categ),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            mode='max',
                            min_delta=0.005,
                            patience=20,
                            restore_best_weights=True
                        )
                    ],
                    verbose=1)


        gate_models = tuner.get_best_models(num_models=num_models_to_keep)
        hyperparameters = tuner.get_best_hyperparameters(num_trials=num_models_to_keep)

        for i, model in enumerate(gate_models):
            logger.info(f"Model {i} summary:")
            logger.info(f"{model.summary()}")
            logger.info(f"Hyperparameters for model {i}: {hyperparameters[i].values}")

            test_loss, test_accuracy = model.evaluate(x_test, y_test_categ, verbose=0)
            logger.info(f"Test accuracy from tuned model number {i}: {test_accuracy:.4f}")

            learning_rate = hyperparameters[i].values.get('learning_rate', 0.002)
            gate_layer = next(l for l in model.layers if hasattr(l, 'get_gate_values'))
            gate_values = gate_layer.get_gate_values()
            neurons_used = [l.units for l in model.layers if hasattr(l, 'units') and 'hidden_dense' in l.name]
            logger.info(f"Gate values for model {i}: {gate_values}")
            logger.info(f"Neurons used for model {i}: {neurons_used}")

            # save unpruned results to csv file
            selected_features = np.arange(num_features)  # essentially all features
            results.append({
                'fold': fold,
                'trial': i,
                'accuracy': test_accuracy,
                **hyperparameters[i].values,
                'threshold': 0.0,
                'features': selected_features.tolist(),
                'cost': np.sum(feature_costs[selected_features]),
            })
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, 'gated_model_results.csv'), index=False)

            # save the model without pruning
            model_name = f"gate_model_fold{fold}_unpruned_trial{i}.keras"
            model.save(os.path.join(results_clf_dir, model_name))

            for thresh in thresholds:
                selected_features = np.where(gate_values > thresh)[0]
                if len(selected_features) == 0:
                    continue

                x_train_sub = x_train[:, selected_features]
                x_test_sub = x_test[:, selected_features]
                cost_sub = feature_costs[selected_features]

                clf = FCNNKerasWrapper(
                    accuracy_metric=AccuracyMetric.Accuracy,
                    tune=False,
                    feature_costs=cost_sub,
                    lambda_reg=None,
                    num_nodes=neurons_used,
                    learning_rate=learning_rate,
                    num_features=len(selected_features),
                    num_classes=num_classes,
                    test_data=(x_test_sub, y_test)
                )
                acc = clf.train(x_train_sub, y_train, x_test_sub, y_test)

                logger.info(f"[PRUNED] Threshold: {thresh:.3f} | Features: {len(selected_features)} | "
                        f"Cost: {np.sum(cost_sub):.2f} | Accuracy: {acc:.4f} | "
                        f"Neurons: {neurons_used}")

                # save results to csv file
                results.append({
                    'fold': fold,
                    'trial': i,
                    'accuracy': acc,
                    **hyperparameters[i].values,
                    'threshold': thresh,
                    'features': selected_features.tolist(),
                    'cost': np.sum(cost_sub),
                })
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(results_dir, 'gated_model_results.csv'), index=False)

                # Save the model pruned model and its train/test data
                pruned_model_name = f"gate_model_fold{fold}_pruned_{thresh:.3f}_trial{i}.keras"
                clf._clf.save(os.path.join(results_clf_dir, pruned_model_name))
                np.save(os.path.join(results_data_dir, f"x_train_fold{fold}_pruned_{thresh:.3f}_trial{i}.npy"), x_train_sub)
                np.save(os.path.join(results_data_dir, f"y_train_fold{fold}_pruned_{thresh:.3f}_trial{i}.npy"), y_train)
                np.save(os.path.join(results_data_dir, f"x_test_fold{fold}_pruned_{thresh:.3f}_trial{i}.npy"), x_test_sub)
                np.save(os.path.join(results_data_dir, f"y_test_fold{fold}_pruned_{thresh:.3f}_trial{i}.npy"), y_test)


if __name__ == "__main__":

    model_path = '/home/balaskas/flexaf/logs/diff_fs_fcnn_wesad___2025.08.06-14.56.18.357/results/classifiers/gate_model_fold0_pruned_0.500_trial0.keras'

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'ConcreteGate': ConcreteGate
        }
    )
    model.summary()
    gate_values = model.get_layer('input_gate_layer').get_gate_values()
    print(f"Gate values: {gate_values}")