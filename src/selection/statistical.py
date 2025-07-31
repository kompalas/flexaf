import logging
import os
import pandas as pd
import numpy as np
from skfeature.function.information_theoretical_based import JMI, DISR
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import f_classif
from src.selection import prepare_feature_data
from src.args import classifier_type_arg, AccuracyMetric, ClassifierType
from src.classifier import set_extra_clf_params, get_classifier
from src.hw_templates.utils import classifier_hw_evaluation

logger = logging.getLogger(__name__)


def jmi_select(X, Y, k):
    fs = JMI.jmi(X, Y)
    return fs[:k].tolist()

def disr_select(X, Y, k):
    fs = DISR.disr(X, Y)
    return fs[:k].tolist()

def fisher_select(X, Y, k):
    # fs = fisher_score.fisher_score(x_train, y_train)
    f_scores, _ = f_classif(X, Y)
    fs = np.argsort(f_scores)[::-1]
    return fs[:k].tolist()


def prune_l1_norm(model, sparsity_ratio):
    for i, W in enumerate(model.coefs_):
        l1_norms = np.abs(W).sum(axis=1)  # L1 norm of each neuron's incoming weights
        threshold = np.percentile(l1_norms, sparsity_ratio * 100)
        mask = l1_norms >= threshold
        # Zero out rows (neurons) with L1 norm below threshold
        model.coefs_[i][~mask, :] = 0
    return model


def prune_l2_norm(model, sparsity_ratio):
    for i, W in enumerate(model.coefs_):
        l2_norms = np.linalg.norm(W, ord=2, axis=1)
        threshold = np.percentile(l2_norms, sparsity_ratio * 100)
        mask = l2_norms >= threshold
        # Zero out rows (neurons) with L2 norm below threshold
        model.coefs_[i][~mask, :] = 0
    return model


def prune_quantize_mlp(classifier, sparsity_levels, input_precision, weight_precisions, test_data, hw_eval_dir, prefix=''):
    """Prune and quantize a multi-layer perceptron classifier."""
    prefix = prefix + '_' if prefix != '' else ''
    results = []
    for sparsity in sparsity_levels:
        # pruned_model = prune_l1_norm(model, sparsity)
        pruned_model = prune_l2_norm(classifier._clf, sparsity)
        classifier._clf = pruned_model
        logger.info(f"Pruned model with sparsity {sparsity}")

        _results = quantize_classifier(classifier, input_precision, weight_precisions, test_data, hw_eval_dir, prefix=f'{prefix}{int(100 * sparsity)}l2norm')
        _results = [r | {'sparsity': sparsity} for r in _results]
        results.extend(_results)
    return results


def quantize_classifier(classifier, input_precision, weight_precisions, test_data, hw_eval_dir, prefix=''):
    """Quantize the classifier's weights and inputs."""
    prefix = prefix + '_' if prefix != '' else ''
    x_test, y_test = test_data

    results = []
    for precision in weight_precisions:
        experiment_name = f'{prefix}{precision}bits'
        this_hw_eval_dir = os.path.join(hw_eval_dir, experiment_name)
        all_inputs_integer = np.all(np.modf(x_test)[0] == 0)
        logger.info(f"Quantizing classifier with {precision}-bit weights and {input_precision}-bit inputs...")

        hw_results, sim_accuracy = classifier_hw_evaluation(
            classifier=classifier,
            test_data=test_data,
            input_precision=input_precision,
            weight_precision=precision,
            savedir=this_hw_eval_dir,
            cleanup=True,
            rescale_inputs=not all_inputs_integer,
            prefix=experiment_name,
            only_rtl=False
        )
        logger.info(f"Quantization accuracy: {sim_accuracy}")
        logger.info(f"Synthesis results: {hw_results._asdict()}")

        results.append(hw_results._asdict() | {
            'input_precision': input_precision,
            'weight_precision': precision,
            'sim_accuracy': sim_accuracy,
        })
    return results


def run_statistical_feature_selection(args):
    """Run an exhaustive search with statistical feature selection methods
    """
    sparsity_levels = [0.2, 0.5, 0.9]
    weight_precisions = [4, 6, 8, 10]
    input_precision = args.default_inputs_precision
    feature_sizes = [5, 10, 15, 20, 25, 30]
    classifiers = ['mlp', 'decisiontree', 'svm']
    feature_selectors = {
        'DISR': disr_select,
        'Fisher': fisher_select,
        'JMI': jmi_select
    }

    # create results directory
    hw_eval_dir = os.path.join(args.resdir, 'hw_eval')
    os.makedirs(hw_eval_dir, exist_ok=True)

    # load dataset and split into train/test
    train_data, test_data, categ_labels, feature_costs, extra_params, _ = prepare_feature_data(args)
    x_train, y_train = train_data
    x_test, y_test = test_data

    all_results = []
    for num_features in feature_sizes:
        for fs_name, selector in feature_selectors.items():

            # perform feature selection
            selected_features = selector(x_train, y_train, k=num_features)
            x_train_sub = x_train[:, selected_features]
            x_test_sub = x_test[:, selected_features]

            for clf_name in classifiers:
                logger.info(f"Running {fs_name} with {clf_name} on {num_features} features...")

                # train floating-point classifier
                clf_type = classifier_type_arg(clf_name)
                extra_params = set_extra_clf_params(clf_type)
                clf = get_classifier(clf_type,
                                     accuracy_metric=AccuracyMetric.Accuracy,
                                     tune=True,
                                     train_data=(x_train_sub, y_train),
                                     seed=args.global_seed,
                                     **extra_params)
                fp_accuracy = clf.train(x_train_sub, y_train, x_test_sub, y_test)
                logger.info(f"Floating-point accuracy: {fp_accuracy}")

                if clf_type == ClassifierType.MLP:
                    results = prune_quantize_mlp(clf, sparsity_levels, input_precision, weight_precisions, (x_test_sub, y_test), hw_eval_dir)
                else:
                    results = quantize_classifier(clf, input_precision, weight_precisions, (x_test_sub, y_test), hw_eval_dir, prefix=f'{num_features}{fs_name}_{clf_name}')
                    results = [r | {'sparsity': 0} for r in results]
                        
                results = [r | {
                    'feature_selector': fs_name,
                    'num_features': num_features,
                    'classifier': clf_name,
                    'fp_accuracy': fp_accuracy
                } for r in results]
                all_results.extend(results)

                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(args.resdir, 'statistical_results.csv'), index=False)

    logger.info("Statistical-based exhaustive search completed and results saved.")
