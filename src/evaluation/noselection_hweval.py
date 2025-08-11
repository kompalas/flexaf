import os
import pickle
import numpy as np
import argparse
import re
from collections import OrderedDict
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from src.hw_templates.utils import classifier_hw_evaluation
from src.args import AccuracyMetric, ClassifierType, dataset_type_arg
from src.classifier import get_classifier
from src.utils import transform_categorical
from src.dataset import get_dataset
from src.features import create_features_from_df_subjectwise
from src.selection import kept_features, all_features


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


def extract_test_data_from_logfile(logfile):
    fold_data = {}
    with open(logfile, 'r') as f:
        for line in f:
            # Match fold subjects line
            m_fold = re.search(r'Running fold (\d+)/\d+ with subjects: train=\[(.*?)\], test=\[(.*?)\]', line)
            if m_fold:
                fold_num = int(m_fold.group(1))
                raw_test = m_fold.group(3).strip()

                if raw_test:
                    test_subjects = []
                    for item in raw_test.split(', '):
                        item = item.strip().strip("'").strip('"')  # remove quotes if present
                        try:
                            test_subjects.append(int(item))  # numeric → int
                        except ValueError:
                            test_subjects.append(item)       # non-numeric → str
                else:
                    test_subjects = []

                fold_data[fold_num] = {'test': test_subjects}

            # Match accuracy line
            m_acc = re.search(r'Accuracy for fold (\d+): ([0-9.]+)', line)
            if m_acc:
                fold_num = int(m_acc.group(1))
                accuracy = float(m_acc.group(2))
                if fold_num in fold_data:
                    fold_data[fold_num]['accuracy'] = accuracy

            # Match best accuracy line
            m_best = re.search(r'Best accuracy: ([0-9.]+) in fold (\d+)', line)
            if m_best:
                best_fold = int(m_best.group(2))

    # Now get the test list for the best fold
    best_test_subjects = fold_data[best_fold]['test']
    return best_test_subjects


def hw_eval_with_all_features(
        resdir, logfile, dataset_type, dataset_file,
        use_all_features=False, 
        weight_precision=8, input_precision=4, neurons=[100], batch_size=32,
        experiment_name=None
    ):
    """Run a single run of statistical feature selection and hardware evaluation."""
    data, sampling_rates, dataset_sr = get_dataset(
        dataset_type, dataset_file,
        resampling_rate=32,
        binary_classification=False,
        three_class_classification=True,
        test_size=None,
    )
    num_sensors = len(data.columns) - 2  # Exclude 'label' and 'subject'
    num_classes = len(data['label'].unique())
    features_dict = OrderedDict([
        (sensor_id, all_features) if use_all_features else
        (sensor_id, kept_features)
        for sensor_id in range(0, num_sensors)
    ])
    print(f"Using features: {features_dict}")

    # Extract test subjects from the logfile and train subjects as the rest
    test_subjects = extract_test_data_from_logfile(logfile)
    print(f"Test subjects extracted from logfile: {test_subjects}")
    train_subjects = data['subject'].unique().tolist()
    train_subjects = [s for s in train_subjects if s not in test_subjects]
    train_data = data[data['subject'].isin(train_subjects)]
    test = data[data['subject'].isin(test_subjects)]
    print(f"Train subjects: {train_subjects}, Test subjects: {test_subjects}")

    x_train, y_train = create_features_from_df_subjectwise(
        data=train_data,
        features_dict=features_dict,
        inputs_precisions=[16] * train_data.shape[1],
        sampling_rates=[dataset_sr] * train_data.shape[1],
        original_sampling_rate=dataset_sr,
        window_size=1,
        target_clock=1
    )
    x_test, y_test = create_features_from_df_subjectwise(
        data=test,
        features_dict=features_dict,
        inputs_precisions=[input_precision] * test.shape[1],
        sampling_rates=[dataset_sr] * test.shape[1],
        original_sampling_rate=dataset_sr,
        window_size=1,
        target_clock=1
    )
    num_features = x_train.shape[1]
    y_train_categ = transform_categorical(y_train, num_classes=num_classes)
    y_test_categ = transform_categorical(y_test, num_classes=num_classes)

    # Train base model and get the accuracy
    model = create_model(num_features, num_classes, neurons)
    model.fit(x_train, y_train_categ, epochs=50, batch_size=batch_size, verbose=0)
    acc = model.evaluate(x_test, y_test_categ, verbose=0)[1]
    print(f"Floating-point accuracy: {acc:.4f}")

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

    experiment_name = f'{os.path.basename(resdir)}' if experiment_name is None else experiment_name
    this_hw_eval_dir = os.path.join(resdir, 'hw_eval', experiment_name)
    all_inputs_integer = np.all(np.modf(x_test)[0] == 0)
    hw_results, sim_accuracy = classifier_hw_evaluation(
        classifier=classifier,
        test_data=(x_test, y_test_categ),
        input_precision=input_precision,
        weight_precision=weight_precision,
        savedir=this_hw_eval_dir,
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
    parser.add_argument('--dataset-type', type=dataset_type_arg, help='Type of dataset to use')
    parser.add_argument('--dataset-file', type=str, help='Path to the dataset file')
    parser.add_argument('--weight-precision', type=int, default=8, help='Weight precision for HW evaluation')
    parser.add_argument('--input-precision', type=int, default=4, help='Input precision for HW evaluation')
    parser.add_argument('--neurons', type=int, nargs='+', default=[100], help='Number of neurons in hidden layers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use-all-features', action='store_true', help='Use all features instead of selected ones')
    args = parser.parse_args()
    expdir = args.expdir
    dataset_type = args.dataset_type
    dataset_file = args.dataset_file
    use_all_features = args.use_all_features
    weight_precision = args.weight_precision
    input_precision = args.input_precision
    neurons = args.neurons
    batch_size = args.batch_size

    # expdir = '/home/balaskas/flexaf/saved_logs/simple_eval/simple_eval_fcnn_spd___2025.08.07-22.15.48.757'

    logfile = os.path.join(expdir, os.path.basename(expdir) + '.log')

    hw_eval_with_all_features(
        resdir=expdir,
        logfile=logfile,
        dataset_type=dataset_type,
        dataset_file=dataset_file,
        use_all_features=use_all_features,
        weight_precision=weight_precision,
        input_precision=input_precision,
        neurons=neurons,
        batch_size=batch_size
    )