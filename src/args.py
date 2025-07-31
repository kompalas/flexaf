import re
import os.path
import argparse
from enum import Enum
from multiprocessing import cpu_count
import warnings


__all__ = [
    'cmd_args', 'ga_args', 'validate_args', 'crossover_param_type', 'GeneType', 'gene_type_arg',
    'ClassifierType', 'classifier_type_arg', 'AccuracyMetric', 'accuracy_metric_arg',
    'HW_Metric', 'hw_metric_arg'
]


def cmd_args(parser):
    """Arguments for running the main application"""
    parser.add_argument('--name', '-n', help='Experiment name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    parser.add_argument('--deterministic', action='store_true', help='Run the application in a deterministic way')
    parser.add_argument('--global-seed', '--seed', type=int, default=123, dest='global_seed',
                         help='Global seed for the application. Used if deterministic is set to True')
    parser.add_argument('--yaml-cfg-file', required=True, help='YAML file containing the experiment description')
    
    app_args = parser.add_argument_group("Problem-specific arguments")

    # Dataset-specific arguments
    app_args.add_argument("--dataset", type=dataset_type_arg, dest='dataset_type', default='wesad_rb',
                          help=f"Specify the dataset. Options: {' | '.join(str_to_dataset_type_map.keys())}")
    app_args.add_argument("--dataset-file", type=str, help="Specify the dataset file path.")
    classification_type_args = app_args.add_mutually_exclusive_group()
    classification_type_args.add_argument("--binary-classification", action='store_true',
                                          help="Enable binary classification (Only for WESAD).")
    classification_type_args.add_argument("--three-class-classification", action='store_true',
                                          help="Enable three-class classification (Only for WESAD).")
    
    # Window size-specific arguments
    app_args.add_argument("--default-window-size", type=int, default=1,
                          help="Specify the default window size for the feature extraction, in seconds. Default is 1 second.")

    # Precision/ADC-specific arguments
    app_args.add_argument("--default-inputs-precision", type=int, default=4,
                          help="Specify the default precision of the inputs. Default is 4.")

    # Classifier-specific arguments
    app_args.add_argument("--classifier-type", type=classifier_type_arg, default='mlp',
                          help="Specify the type of classifier to use.")
    app_args.add_argument("--tune-classifier", action='store_true', help="Enable hyperparameter tuning for the classifier.")
    app_args.add_argument("--test-size", type=float, default=0.3,
                          help="Specify the test size for the train-test split. Default is 0.3.")

    # Optimization/Objective-specific arguments
    app_args.add_argument("--performance-target", type=float, default=1.0,
                          help="Set the real-time performance target for the application, in seconds (frequency of predictions). Default is 1 second.")
    
    # Feature selection-specific arguments
    fs_args = app_args.add_mutually_exclusive_group()
    fs_args.add_argument("--execute-differentiable-feature-selection", action='store_true',
                         help="Enable differentiable feature selection using ConcreteGate.")
    fs_args.add_argument("--execute-heuristic-feature-selection", action='store_true',
                         help="Enable heuristic feature selection.")
    fs_args.add_argument("--execute-greedy-feature-selection", action='store_true',
                         help="Enable greedy feature selection.")
    fs_args.add_argument("--execute-statistical-feature-selection", action='store_true',
                         help="Enable statistical feature selection.")
    return parser


def validate_args(args):
    if not args.deterministic:
        args.global_seed = None


### Enumeration and argument type functions

class ClassifierType(Enum):
    GradientBoosting = 0
    DecisionTree = 1
    DecisionTreeRegressor = 2
    SVM = 3
    MLP = 4
    TNN = 5
    BNN = 6
    FCNN = 7

str_to_classifier_type_map = {
    entry.name.lower(): entry for entry in ClassifierType
}

def classifier_type_arg(classifier_str):
    try:
        return str_to_classifier_type_map[classifier_str.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('--classifier argument must be one of {0} (received {1})'.format(
            list(str_to_classifier_type_map.keys()), classifier_str
        ))


class DatasetType(Enum):
    WESAD_RB = 0
    WESAD_E4 = 1
    WESAD_Merged = 2
    StressInNurses = 3
    SPD = 4
    AffectiveROAD = 5
    DriveDB = 6
    HAR = 7
    HARTH = 8
    WISDM = 9
    Coswara = 10

str_to_dataset_type_map = {entry.name.lower(): entry for entry in DatasetType}

def dataset_type_arg(dataset_str):
    try:
        return str_to_dataset_type_map[dataset_str.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('--dataset argument must be one of {0} (received {1})'.format(
            list(str_to_dataset_type_map.keys()), dataset_str
        ))


class AccuracyMetric(Enum):
    Accuracy = 0
    F1 = 1

str_to_accuracy_metric_map = {
    'accuracy': AccuracyMetric.Accuracy,
    'f1': AccuracyMetric.F1
}

def accuracy_metric_arg(metric_str):
    if metric_str is None:
        return
    try:
        return str_to_accuracy_metric_map[metric_str.replace('_', '').replace('-', '').lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('--accuracy-metric argument must be one of {0} (received {1})'.format(
            list(str_to_accuracy_metric_map.keys()), metric_str
        ))