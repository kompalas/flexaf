import logging
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from qkeras import quantizers
from skfeature.function.information_theoretical_based import JMI, DISR
from sklearn.feature_selection import f_classif
from src.selection import prepare_feature_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------- Feature selection methods ---------------- #
def jmi_select(X, Y, k):
    fs = JMI.jmi(X, Y)
    return fs[:k].tolist()

def disr_select(X, Y, k):
    fs = DISR.disr(X, Y)
    return fs[:k].tolist()

def fisher_select(X, Y, k):
    f_scores, _ = f_classif(X, Y)
    fs = np.argsort(f_scores)[::-1]
    return fs[:k].tolist()


# ---------------- Pruning (in-place) ---------------- #
def prune_l2_norm(model, sparsity_ratio):
    """Prune neurons in-place based on L2 norm of their weights."""
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            weights, biases = layer.get_weights()
            l2_norms = np.linalg.norm(weights, ord=2, axis=1)
            threshold = np.percentile(l2_norms, sparsity_ratio * 100)
            mask = l2_norms >= threshold
            weights[~mask, :] = 0
            layer.set_weights([weights, biases])


# ---------------- Quantization (in-place) ---------------- #
def quantize_model(model, precision):
    """Apply simulated quantization in-place."""
    if precision == 8:
        # Placeholder for real TFLite evaluation if desired
        return
    else:
        for layer in model.layers:
            if isinstance(layer, layers.Dense):
                weights, biases = layer.get_weights()
                # Simulate quantization by rounding to nearest level
                scale = 2 ** precision - 1
                weights = np.round(weights * scale) / scale
                biases = np.round(biases * scale) / scale
                layer.set_weights([weights, biases])


# ---------------- Model creation ---------------- #
def create_model(num_features, num_classes, neurons, learning_rate=0.001):
    inputs = layers.Input(shape=(num_features,), name='input')
    x = inputs
    for i, units in enumerate(neurons):
        x = layers.Dense(units, activation='relu', name=f"hidden_dense_{i}")(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model    


# ---------------- Main loop ---------------- #
def run_soa_statistical_feature_selection(args):
    sparsity_levels = [0.2, 0.5, 0.9]
    weight_precisions = [4, 6, 8, 10]
    neurons = [100]
    batch_size = 32
    feature_sizes = args.statistical_num_features

    feature_selectors = {
        'DISR': disr_select,
        'Fisher': fisher_select,
        'JMI': jmi_select
    }

    hw_eval_dir = os.path.join(args.resdir, 'hw_eval')
    os.makedirs(hw_eval_dir, exist_ok=True)

    train_data, test_data, categ_labels, feature_costs, params, _ = prepare_feature_data(
        args, use_all_features=args.statistical_use_all_features
    )
    x_train, y_train = train_data
    x_test, y_test = test_data
    y_train_categ, y_test_categ = categ_labels
    num_classes = params['num_classes']

    all_results = []
    results_path = os.path.join(args.resdir, 'statistical_results.csv')

    for k_features in feature_sizes:
        if k_features > x_train.shape[1]:
            logger.warning(f"Skipping {k_features} features (only {x_train.shape[1]} available).")
            continue

        for fs_name, selector in feature_selectors.items():
            logger.info(f"[FS] {fs_name} with {k_features} features...")
            selected_features = selector(x_train, y_train, k=k_features)
            if feature_costs is not None:
                total_cost = np.sum(feature_costs[selected_features])
            else:
                total_cost = 0.0

            x_train_sub = x_train[:, selected_features]
            x_test_sub = x_test[:, selected_features]

            # Train base model
            model = create_model(k_features, num_classes, neurons)
            model.fit(x_train_sub, y_train_categ, epochs=50, batch_size=batch_size, verbose=0)

            # 1) Floating-point accuracy
            acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
            logger.info(f"  Floating-point accuracy: {acc:.4f}")
            all_results.append({
                'fs_method': fs_name,
                'num_features': k_features,
                'total_cost': total_cost,
                'sparsity': 0.0,
                'precision': 32,
                'accuracy': acc
            })
            pd.DataFrame(all_results).to_csv(results_path, index=False)
            model.save(os.path.join(hw_eval_dir, f"{fs_name}_{k_features}_float.keras"))

            # 2) Pruning and quantization
            for sparsity in sparsity_levels:
                
                # Prune and fine-tune the pruned model for a few epochs to recover accuracy
                prune_l2_norm(model, sparsity)
                model.fit(x_train_sub, y_train_categ, epochs=10, batch_size=batch_size, verbose=0)
                pruned_acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
                logger.info(f"  Pruned (sparsity={sparsity}) accuracy: {pruned_acc:.4f}")

                all_results.append({
                    'fs_method': fs_name,
                    'num_features': k_features,
                    'total_cost': total_cost,
                    'sparsity': sparsity,
                    'precision': 32,
                    'accuracy': pruned_acc
                })
                pd.DataFrame(all_results).to_csv(results_path, index=False)
                model.save(os.path.join(hw_eval_dir, f"{fs_name}_{k_features}_pruned_{sparsity}.keras"))

                for precision in weight_precisions:
                    quantize_model(model, precision)
                    q_acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
                    logger.info(f"  Quantized (precision={precision} bits) accuracy: {q_acc:.4f}")
                    all_results.append({
                        'fs_method': fs_name,
                        'num_features': k_features,
                        'total_cost': total_cost,
                        'sparsity': sparsity,
                        'precision': precision,
                        'accuracy': q_acc
                    })
                    pd.DataFrame(all_results).to_csv(results_path, index=False)
                    model.save(os.path.join(hw_eval_dir, f"{fs_name}_{k_features}_pruned_{sparsity}_q{precision}.keras"))

    logger.info("Feature selection and model evaluation completed.")
    logger.info(f"Results saved to {results_path}")
