import numpy as np
from src.hw_templates.mlp2verilog import write_mlp_to_verilog


class KerasAsSklearnMLP:
    def __init__(self, keras_model, classes=None):
        self.model = keras_model
        self.coefs_, self.intercepts_ = self._extract_weights()
        self.activation = self._map_activation()
        self.classes_ = np.arange(self.coefs_[-1].shape[1]) if classes is None else classes

    def _extract_weights(self):
        weights = self.model.get_weights()
        coefs = []
        intercepts = []
        for i in range(0, len(weights), 2):
            coefs.append(weights[i])
            intercepts.append(weights[i + 1])
        return coefs, intercepts

    def _map_activation(self):
        # assumes all hidden layers use same activation
        first_hidden_layer = [layer for layer in self.model.layers if hasattr(layer, 'activation')][0]
        act = first_hidden_layer.activation.__name__
        if act in ['relu', 'tanh', 'logistic', 'linear']:
            return act
        elif act == 'sigmoid':
            return 'logistic'
        else:
            raise ValueError(f"Unsupported activation: {act}")


def write_keras_model_to_verilog(keras_model, input_bits, weight_bits, verilog_file, tb_file, inputs_file, output_file, simclk_ms=1):
    """ Convert a Keras model to a Verilog representation."""
    # Wrap the Keras model as a scikit-learn compatible MLP
    mlp_compatible_model = KerasAsSklearnMLP(keras_model)

    # Use the existing function to write the model to Verilog
    metadata = write_mlp_to_verilog(
        mlp_model=mlp_compatible_model,
        input_bits=input_bits,
        weight_bits=weight_bits,
        verilog_file=verilog_file,
        tb_file=tb_file,
        inputs_file=inputs_file,
        output_file=output_file,
        input_separator=' ',
        simclk_ms=simclk_ms
    )
    return metadata


if __name__ == "__main__":

    import os
    import numpy as np
    import tensorflow as tf
    from src.utils import transform_categorical
    from src.classifier import get_classifier, set_extra_clf_params
    from src.args import ClassifierType, AccuracyMetric
    from src.hw_templates.utils import classifier_hw_evaluation

    # load the saved model and test set and evaluate its accuracy
    expdir = '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.06-16.16.48.524'
    fold = 0
    sparsity = 0.5
    trial = 2
    
    model_name = f'gate_model_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.keras'

    model = tf.keras.models.load_model(
        os.path.join(expdir, 'results', 'classifiers', model_name),
    )
    x_test_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_test_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_test = np.load(os.path.join(expdir, 'results', 'data', f'y_test_fold{fold}.npy'))
    y_test_categ = transform_categorical(y_test, num_classes=model.output_shape[-1])
    x_train_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_train_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_train = np.load(os.path.join(expdir, 'results', 'data', f'y_train_fold{fold}.npy'))
    y_train_categ = transform_categorical(y_train, num_classes=model.output_shape[-1])

    this_hw_eval_dir = os.path.join(expdir, 'hw_eval')
    os.makedirs(this_hw_eval_dir, exist_ok=True)

    extra_params = set_extra_clf_params(
        ClassifierType.FCNN,
        input_precisions=[4] * x_test_sub.shape[1],
        x_test=x_test_sub, y_test=y_test,
        feature_costs=None  # No feature costs needed for evaluation
    )
    classifier = get_classifier(
        ClassifierType.FCNN,
        accuracy_metric=AccuracyMetric.Accuracy,
        tune=False,
        train_data=None,  # No training data needed for evaluation
        seed=42,
        **extra_params
    )
    classifier._clf = model  # Set the pre-trained model as the classifier

    hw_results, sim_accuracy = classifier_hw_evaluation(
        classifier=classifier,
        test_data=(x_test_sub, y_test_categ),
        input_precision=4,
        weight_precision=8,
        savedir=this_hw_eval_dir,
        cleanup=True,
        rescale_inputs=False,
        prefix=os.path.basename(expdir),
        only_rtl=True
    )
    print(f"Quantization accuracy: {sim_accuracy}")
    print(f"Synthesis results: {hw_results._asdict()}")
