import argparse
import os
import numpy as np
import tensorflow as tf
from src.hw_templates.utils import classifier_hw_evaluation
from src.args import AccuracyMetric, ClassifierType
from src.classifier import get_classifier
from src.utils import transform_categorical


def load_gates_experiment_model_data(expdir, fold, sparsity, trial):
    model_name = f'gate_model_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.keras'
    model = tf.keras.models.load_model(
        os.path.join(expdir, 'results', 'classifiers', model_name),
    )
    x_test_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_test_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_test = np.load(os.path.join(expdir, 'results', 'data', f'y_test_fold{fold}.npy'))
    y_test_categ = transform_categorical(y_test, num_classes=model.output_shape[-1])
    return model, x_test_sub, y_test_categ


def hw_eval_with_single_model(
        resdir, model, x_test, y_test_categ, weight_precision=8, input_precision=4, experiment_name='single_model_eval'
    ):
    """Run a single hardware evaluation on a given Keras NN."""
    num_features = x_test.shape[1]
    num_classes = model.output_shape[-1]

    # transform fully connected keras model to MLP
    classifier = get_classifier(
        ClassifierType.FCNN,
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=False,
        train_data=None,
        seed=None,
        input_precisions=[input_precision] * x_test.shape[1],
        x_test=x_test, y_test=y_test_categ,
        feature_costs=None,
        num_features=num_features,
        num_classes=num_classes,
    )
    classifier._clf = model

    this_hw_eval_dir = os.path.join(resdir, 'hw_eval', experiment_name)
    all_inputs_integer = np.all(np.modf(x_test)[0] == 0)
    hw_results, sim_accuracy = classifier_hw_evaluation(
        classifier=classifier,
        test_data=(x_test, y_test_categ),
        input_precision=input_precision,
        weight_precision=weight_precision,
        savedir=this_hw_eval_dir,
        simclk_ms=1,
        cleanup=True,
        rescale_inputs=not all_inputs_integer,
        prefix=experiment_name,
        only_rtl=False
    )
    print(f"Simulation accuracy: {sim_accuracy:.4f}")
    print(f"Synthesis results: {hw_results._asdict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical feature selection and HW evaluation")
    parser.add_argument('--expdir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--weight-precision', type=int, default=8, help='Weight precision for HW evaluation')
    parser.add_argument('--input-precision', type=int, default=4, help='Input precision for HW evaluation')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for the experiment')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity value for the model')
    parser.add_argument('--trial', type=int, default=2, help='Trial number for the experiment')
    args = parser.parse_args()
    expdir = args.expdir
    fold = args.fold
    sparsity = args.sparsity
    trial = args.trial

    # expdir = '/home/balaskas/flexaf/saved_logs/spd/diff_fs_fcnn_spd___2025.08.06-20.55.49.900'
    # lottery_ticket = False
    # fold = 0
    # sparsity = 0.5
    # trial = 2

    model, x_test, y_test_categ = load_gates_experiment_model_data(expdir, fold, sparsity, trial)

    hw_eval_with_single_model(
        resdir=expdir,
        model=model,
        x_test=x_test,
        y_test_categ=y_test_categ,
        weight_precision=8,
        input_precision=4,
        experiment_name='single_model_eval'
    )