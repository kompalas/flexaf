import logging
from collections import OrderedDict
import numpy as np
from src.selection import feature_costs_map, kept_features, prepare_feature_data_cross_validation
from src.features import create_features_from_df_subjectwise
from src.classifier import set_extra_clf_params, get_classifier
from src.args import AccuracyMetric
from src.utils import transform_categorical


logger = logging.getLogger(__name__)



def perform_basic_evaluation(args):
    """Perform a simple evaluation of the dataset with basic features and k-fold cross-validation.
    """
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
        x_train = x_train.values.astype(np.float32)
        x_test = x_test.values.astype(np.float32)

        extra_params = set_extra_clf_params(
            args.classifier_type,
            input_precisions=input_precisions,
            x_test=x_test, y_test=y_test,
            feature_costs=None
        )
        if 'num_classes' in extra_params:
            extra_params['num_classes'] = num_classes

        classifier = get_classifier(
            args.classifier_type,
            accuracy_metric=AccuracyMetric.Accuracy,
            tune=args.tune_classifier,
            train_data=(x_train, y_train),
            seed=args.global_seed,
            **extra_params
        )
        accuracy = classifier.train(x_train, y_train, x_test, y_test)
        results.append(accuracy)
        logger.info(f"Accuracy for fold {fold + 1}: {accuracy:.4f}")

    fold_w_highest_accuracy = np.argmax(results)
    best_accuracy = results[fold_w_highest_accuracy]
    logger.info(f"Best accuracy: {best_accuracy:.4f} in fold {fold_w_highest_accuracy + 1}")
