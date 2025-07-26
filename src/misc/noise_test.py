import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple
from copy import deepcopy


def load_data():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=42)


def add_relative_noise(data: np.ndarray, noise_percent=0.1, distribution='uniform') -> np.ndarray:
    """
    Add value-dependent (multiplicative) noise to simulate sensor digitization effects (e.g. ADC noise).

    Parameters:
    - data (np.ndarray): Input array of sensor readings or any measurements.
    - noise_percent (float): Relative noise level (e.g. 0.1 means ±10%).
    - distribution (str): Type of distribution ('uniform' or 'normal').

    Returns:
    - np.ndarray: Noisy array where noise scales with the data.
    """
    if distribution == 'uniform':
        noise_factor = np.random.uniform(low=1 - noise_percent,
                                         high=1 + noise_percent,
                                         size=data.shape)
    elif distribution == 'normal':
        noise_factor = np.random.normal(loc=1.0,
                                        scale=noise_percent,
                                        size=data.shape)
    else:
        raise ValueError("Unsupported distribution. Use 'uniform' or 'normal'.")

    return data * noise_factor


def get_classifier(model_type: str):
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif model_type == 'mlp':
        return MLPClassifier(random_state=42, max_iter=1000)
    else:
        raise ValueError("Unsupported model type. Choose from 'decision_tree', 'random_forest', 'mlp'.")


def train_and_evaluate(model_type: str, noise_percent=0.1, distribution='uniform'):
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train baseline model on clean data
    clf_baseline = get_classifier(model_type)
    clf_baseline.fit(X_train, y_train)

    # Evaluate on clean test data
    baseline_clean_acc = accuracy_score(y_test, clf_baseline.predict(X_test))

    # Evaluate baseline model on noisy test data
    X_test_noisy = add_relative_noise(X_test, noise_percent=noise_percent, distribution=distribution)
    baseline_noisy_acc = accuracy_score(y_test, clf_baseline.predict(X_test_noisy))

    # Data augmentation (append noisy training data)
    X_train_noisy = add_relative_noise(X_train, noise_percent=noise_percent, distribution=distribution)
    X_augmented = np.vstack([X_train, X_train_noisy])
    y_augmented = np.hstack([y_train, y_train])

    # Retrain on augmented data
    clf_augmented = get_classifier(model_type)
    clf_augmented.fit(X_augmented, y_augmented)

    # Evaluate on noisy test data
    augmented_noisy_acc = accuracy_score(y_test, clf_augmented.predict(X_test_noisy))

    # Report results
    print(f"Model type: {model_type}")
    print(f"Baseline accuracy (clean test): {baseline_clean_acc:.4f}")
    print(f"Baseline accuracy (noisy test): {baseline_noisy_acc:.4f}")
    print(f"Augmented model accuracy (noisy test): {augmented_noisy_acc:.4f}")


if __name__ == "__main__":
    for model in ['decision_tree', 'random_forest', 'mlp']:
        train_and_evaluate(model_type=model)
