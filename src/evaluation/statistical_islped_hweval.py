import logging
import os
import pickle
import numpy as np
import argparse
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from skfeature.function.information_theoretical_based import JMI, DISR
from sklearn.feature_selection import f_classif
from src.selection import prepare_feature_data
from src.hw_templates.utils import classifier_hw_evaluation
from src.args import AccuracyMetric, ClassifierType
from src.classifier import get_classifier
from src.hw_templates.keras2verilog import KerasAsSklearnMLP


def setup_logger(log_path):
    logger = logging.getLogger("ISLPED_HW_Eval")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Prevent propagation to root logger (fixes console duplication!)
    logger.propagate = False

    # File formatter (rich detail)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] (%(processName)s) %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S"
    )

    # Console formatter (simple)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)

    # Attach handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

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
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model    


# ---------------- Main loop ---------------- #
def run_single_statistical_feature_selection_and_hw_eval(
        exp_args, resdir,
        sparsity=0.1, weight_precision=8, input_precision=4, neurons=[100], batch_size=32,
        num_features=10, feature_selector_name='fisher', use_all_features=False, 
        logger=None
    ):
    """Run a single run of statistical feature selection and hardware evaluation."""
    feature_selector = {
        'DISR': disr_select,
        'Fisher': fisher_select,
        'JMI': jmi_select
    }.get(feature_selector_name)

    # Load the dataset and prepare the data
    train_data, test_data, categ_labels, feature_costs, params, _ = prepare_feature_data(
        exp_args, use_all_features=use_all_features
    )
    x_train, y_train = train_data
    x_test, y_test = test_data
    y_train_categ, y_test_categ = categ_labels
    num_classes = params['num_classes']

    # Select features using the chosen feature selector
    selected_features = feature_selector(x_train, y_train, k=num_features)
    if feature_costs is not None:
        total_cost = np.sum(feature_costs[selected_features])
    else:
        total_cost = 0.0
    if num_features > x_train.shape[1]:
        num_features = x_train.shape[1]
        x_train_sub = x_train
        x_test_sub = x_test
        logger.info(f"Warning: num_features ({num_features}) exceeds available features ({x_train.shape[1]}), using all available features.")
    else:
        x_train_sub = x_train[:, selected_features]
        x_test_sub = x_test[:, selected_features]
    logger.info(f"Selected features: {selected_features}, Total cost: {total_cost}")

    # load the model
    pruning_name = f'pruned_{sparsity}' if sparsity > 0.01 else 'float'
    quant_name = f'_q{weight_precision}' if weight_precision < 32 else ''
    model_name = f'{feature_selector_name}_{num_features}_{pruning_name}{quant_name}.keras'
    model_path = os.path.join(resdir, 'hw_eval', model_name)
    assert os.path.exists(model_path), f"Model file {model_path} does not exist."
    model = models.load_model(model_path)

    # # Train base model
    # model = create_model(num_features, num_classes, neurons)
    # model.fit(x_train_sub, y_train_categ, epochs=50, batch_size=batch_size, verbose=0)

    # # 1) Floating-point accuracy
    # acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
    # logger.info(f"Floating-point accuracy: {acc:.4f}")

    # if sparsity > 0.01:

    #     # Prune and fine-tune the pruned model for a few epochs to recover accuracy
    #     prune_l2_norm(model, sparsity)
    #     model.fit(x_train_sub, y_train_categ, epochs=10, batch_size=batch_size, verbose=0)
    #     pruned_acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
    #     logger.info(f"Pruned (sparsity={sparsity}) accuracy: {pruned_acc:.4f}")

    #     if weight_precision < 32:
    #         # 2) Quantized accuracy
    #         quantize_model(model, weight_precision)
    #         q_acc = model.evaluate(x_test_sub, y_test_categ, verbose=0)[1]
    #         logger.info(f"Quantized (precision={weight_precision} bits) accuracy: {q_acc:.4f}")


    # transform fully connected keras model to MLP
    classifier = get_classifier(
        ClassifierType.FCNN,
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=False,
        train_data=None,
        seed=exp_args.global_seed,
        input_precisions=[input_precision] * x_test_sub.shape[1],
        x_test=x_test_sub, y_test=y_test_categ,
        feature_costs=None,
        num_features=x_test_sub.shape[1],
        num_classes=num_classes,
    )
    classifier._clf = model

    experiment_name = f'{os.path.basename(resdir)}_{feature_selector_name}_{num_features}feat_sparsity{int(sparsity*100)}_weightprec{weight_precision}_inputprec{input_precision}'
    this_hw_eval_dir = os.path.join(resdir, 'hw_eval', experiment_name)
    all_inputs_integer = np.all(np.modf(x_test)[0] == 0)
    hw_results, sim_accuracy = classifier_hw_evaluation(
        classifier=classifier,
        test_data=(x_test_sub, y_test_categ),
        input_precision=input_precision,
        weight_precision=weight_precision,
        savedir=this_hw_eval_dir,
        cleanup=True,
        rescale_inputs=not all_inputs_integer,
        prefix=experiment_name,
        only_rtl=False
    )
    logger.info(f"Simulation accuracy: {sim_accuracy:.4f}")
    logger.info(f"Synthesis results: {hw_results._asdict()}")


if __name__ == "__main__":

    # expdir = '/home/balaskas/flexaf/saved_logs/spd/sota_fcnn_spd_allfeat___2025.08.10-02.55.53.940'
    # selector_name = 'Fisher'
    # num_features = 10
    # sparsity = 0.5
    # weight_precision = 32
    # input_precision = 4
    # neurons = [100]
    # batch_size = 32
    # use_all_features = False

    parser = argparse.ArgumentParser(description="Statistical feature selection and HW evaluation")
    parser.add_argument('--expdir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--selector-name', type=str, default='Fisher', choices=['Fisher', 'DISR', 'JMI'], help='Feature selector name')
    parser.add_argument('--num-features', type=int, default=10, help='Number of features to select')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity ratio for pruning')
    parser.add_argument('--weight-precision', type=int, default=32, help='Weight precision (bits)')
    parser.add_argument('--input-precision', type=int, default=4, help='Input precision (bits)')
    parser.add_argument('--neurons', type=int, nargs='+', default=[100], help='List of neurons per layer')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use-all-features', action='store_true', help='Use all features (ignore selection)')

    args = parser.parse_args()

    expdir = args.expdir
    selector_name = args.selector_name
    num_features = args.num_features
    sparsity = args.sparsity
    weight_precision = args.weight_precision
    input_precision = args.input_precision
    neurons = args.neurons
    batch_size = args.batch_size
    use_all_features = args.use_all_features

    try:
        with open(os.path.join(expdir, 'args.yaml'), 'rb') as f:
            exp_args = pickle.load(f)
    except FileNotFoundError:
        with open(os.path.join(expdir, 'args.pkl'), 'rb') as f:
            exp_args = pickle.load(f)

    logger = setup_logger(os.path.join(expdir, 'statistical_hw_eval.log'))
    assert os.path.exists(os.path.join(expdir, 'statistical_hw_eval.log')), "Log file not created."

    run_single_statistical_feature_selection_and_hw_eval(
        exp_args=exp_args,
        resdir=expdir,
        sparsity=sparsity,
        weight_precision=weight_precision,
        input_precision=input_precision,
        neurons=neurons,
        num_features=num_features,
        feature_selector_name=selector_name,
        use_all_features=use_all_features,
        batch_size=batch_size,
        logger=logger
    )