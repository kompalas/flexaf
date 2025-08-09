import logging
import os
import pandas as pd
import numpy as np
from skfeature.function.information_theoretical_based import JMI, DISR
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import f_classif
from src.selection import prepare_feature_data, feature_costs_map
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


def run_statistical_feature_selection(args):
    """Run an exhaustive search with statistical feature selection methods
    """
    input_precision = args.default_inputs_precision
    feature_sizes = args.statistical_num_features
    classifiers = [args.classifier_type.name.lower()]  #['mlp', 'decisiontree', 'svm']
    feature_selectors = {
        # 'DISR': disr_select,
        'Fisher': fisher_select,
        # 'JMI': jmi_select
    }

    # create results directory
    hw_eval_dir = os.path.join(args.resdir, 'hw_eval')
    os.makedirs(hw_eval_dir, exist_ok=True)

    # load dataset and split into train/test
    train_data, test_data, categ_labels, feature_costs, filtered_params, _ = prepare_feature_data(args, use_all_features=True)
    x_train, y_train = train_data
    x_test, y_test = test_data

    for num_features in feature_sizes:
        if num_features > x_train.shape[1]:
            logger.warning(f"Requested {num_features} features, but only {x_train.shape[1]} available. Skipping this size.")
            continue
        for fs_name, selector in feature_selectors.items():

            # perform feature selection
            selected_features = selector(x_train, y_train, k=num_features)
            if feature_costs is not None:
                total_cost = np.sum(feature_costs[selected_features])
                logger.info(f"Selected {num_features} features with total cost {total_cost}")
            else:
                logger.info(f"Selected {num_features} features ({selected_features})")

            x_train_sub = x_train[:, selected_features]
            x_test_sub = x_test[:, selected_features]

            for clf_name in classifiers:
                logger.info(f"Running {fs_name} with {clf_name} on {num_features} features...")

                # train floating-point classifier
                clf_type = classifier_type_arg(clf_name)
                extra_params = set_extra_clf_params(clf_type,
                                                    input_precisions=[input_precision] * x_train_sub.shape[1], 
                                                    x_test=x_test_sub, y_test=y_test)

                train_kwargs = {}
                if clf_type == ClassifierType.FCNN:
                    extra_params['num_classes'] = filtered_params['num_classes']
                    train_kwargs = {'verbose': 1, 'epochs': 50, 'batch_size': 32}

                clf = get_classifier(clf_type,
                                     accuracy_metric=AccuracyMetric.Accuracy,
                                     tune=True,
                                     train_data=(x_train_sub, y_train),
                                     seed=args.global_seed,
                                     **extra_params)
                fp_accuracy = clf.train(x_train_sub, y_train,
                                        x_test_sub, y_test,
                                        **train_kwargs)
                logger.info(f"Floating-point accuracy: {fp_accuracy}")

    logger.info("Statistical-based exhaustive search completed and results saved.")
