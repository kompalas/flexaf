import logging
import os.path
import numpy as np
import pickle
import re
import keras
import keras_tuner
from copy import deepcopy
from functools import partial
from qkeras import QDense, QActivation
from qkeras import utils as qutils
from keras import layers, Sequential, optimizers
from scikeras.wrappers import KerasClassifier
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from src.args import ClassifierType, AccuracyMetric
from src.utils import accuracy_keras, f1score_keras, transform_categorical, convert_to_fixed_point
from src.custom_models.mlp.input_gate_layer import InputFeatureGates
from src.hw_templates.dt2verilog import write_tree_to_verilog
from src.hw_templates.mlp2verilog import write_mlp_to_verilog
# from src.hw_templates.mlp2verilog_bkp import write_mlp_to_verilog
from src.hw_templates.svm2verilog import write_svm_to_verilog
from src.hw_templates.utils import get_maxabs, get_width


logger = logging.getLogger(__name__)


def get_classifier(classifier_type, accuracy_metric, tune=False, train_data=None, seed=None, **extra_clf_params):
    classifiers = {
        ClassifierType.SVM: SVCWrapper,
        ClassifierType.MLP: MLPClassifierWrapper,
        # ClassifierType.CNN: cnn,
        ClassifierType.DecisionTree: DecisionTreeClassifierWrapper,
        ClassifierType.DecisionTreeRegressor: DecisionTreeRegressorWrapper,
        ClassifierType.GradientBoosting: GradientBoostingClassifierWrapper,
        # ClassifierType.TNN: TNNWrapper,
        ClassifierType.TNN: TNNKerasWrapper,
        # ClassifierType.BNN: BNNWrapper,
        ClassifierType.BNN: BNNKerasWrapper,
        ClassifierType.FCNN: FCNNKerasWrapper
    }
    try:
        return classifiers[classifier_type](accuracy_metric, tune, train_data, seed, **extra_clf_params)
    except KeyError:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def set_extra_clf_params(classifier_type, adc_precisions=None, x_test=None, y_test=None, feature_costs=None):
    """Set the extra parameters for the classifier class instantiation.
    """
    extra_params = {}

    if classifier_type in (ClassifierType.BNN, ClassifierType.TNN, ClassifierType.FCNN):
        assert adc_precisions is not None, "For Keras models, the ADC precisions must be provided for the input bitwidth."
        assert y_test is not None and x_test is not None, "For Keras models, the test data must be provided to set the number of classes, features, and samples."
        
        extra_params['input_bitwidth'] = max(adc_precisions)
        extra_params['num_nodes'] = 100
        extra_params['learning_rate'] = 0.1
        extra_params['feature_costs'] = feature_costs
        extra_params['lambda_reg'] = 0.0005

        if x_test is not None and y_test is not None:
            extra_params['num_classes'] = len(np.unique(y_test))
            extra_params['num_features'] = x_test.shape[1]
            extra_params['num_samples'] = x_test.shape[0]
            extra_params['test_data'] = (x_test, y_test)

    elif classifier_type == ClassifierType.DecisionTree:
        extra_params['criterion'] = 'entropy'
        extra_params['max_depth'] = 7

    elif classifier_type == ClassifierType.SVM:
        # extra_params['kernel'] = 'linear'  # no need if the SVC class is not used
        # extra_params['probability'] = True  # this slows down training by A LOT
        extra_params['multi_class'] = 'ovr'  # one-vs-rest for the LinearSVC
        # extra_params['decision_function_shape'] = 'ovo'  # one-vs-one/one-vs-rest for the SVC

    elif classifier_type == ClassifierType.MLP:
        extra_params['activation'] = 'relu'
        extra_params['max_iter'] = 400
        extra_params['hidden_layer_sizes'] = (50,)

    return extra_params


class _ClassifierWrapper:
    # Base class for classifier wrappers
    def __init__(self, accuracy_metric, tune=False, train_data=None, seed=None, **extra_clf_params):
        self._clf = None
        self.set_clf(seed, **extra_clf_params)
        self.tuned_parameters = None
        self.set_tuned_parameters()
        self.set_accuracy_function(accuracy_metric)

        if tune and train_data is not None:
            self._clf = self.tune(train_data, **extra_clf_params)

    def set_clf(self, *args, **extra_clf_params):
        raise NotImplementedError

    def set_tuned_parameters(self, *args, **kwargs):
        pass

    def set_accuracy_function(self, accuracy_metric):
        self.accuracy_metric = accuracy_metric
        self.accuracy_params = {}
        if accuracy_metric == AccuracyMetric.Accuracy:
            self.accuracy_f = metrics.accuracy_score
        elif accuracy_metric == AccuracyMetric.F1:
            self.accuracy_f = metrics.f1_score
            self.accuracy_params['average'] = 'weighted'
        else:
            raise ValueError(f"Unknown accuracy metric: {accuracy_metric}")

    def tune(self, train_data, **extra_clf_params):
        raise NotImplementedError

    def train(self, x_train, y_train, x_test=None, y_test=None, **train_kwargs):
        self._clf.fit(x_train, y_train, **train_kwargs)
        if x_test is None or y_test is None:
            return
        return self.test(x_test, y_test)

    def test(self, x_test, y_test):
        y_pred = self._clf.predict(x_test)
        accuracy = self.accuracy_f(y_test, y_pred, **self.accuracy_params)
        return accuracy

    def get_weights(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_architecture(self, *args, **kwargs):
        raise NotImplementedError

    def save_weights(self, *args, **kwargs):
        raise NotImplementedError
    
    def to_verilog(self, *args, **kwargs):
        raise NotImplementedError
    
    def fixed_point_inference(self, *args, **kwargs):
        raise NotImplementedError


class SKLearnClassifierWrapper(_ClassifierWrapper):
    def tune(self, train_data, **extra_clf_params):
        if self.tuned_parameters is None:
            return self._clf

        # remove parameters that are included for tuning
        for key in self.tuned_parameters.keys():
            extra_clf_params.pop(key, None)

        grid = GridSearchCV(self._clf, self.tuned_parameters, refit=True, verbose=0)
        grid.fit(*train_data)
        best_params = grid.best_params_
        classifier_class = self._clf.__class__
        return classifier_class(**best_params, **extra_clf_params)


class GradientBoostingClassifierWrapper(SKLearnClassifierWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        self._clf = GradientBoostingClassifier(random_state=seed, **extra_clf_params)

    def set_tuned_parameters(self):
        self.tuned_parameters = {'n_estimators': [50, 100, 150],
                                 'learning_rate': [0.01, 0.1, 0.2],
                                 'max_depth': [3, 4, 5]}

    def get_architecture(self):
        return self._clf.estimators_.shape

    def get_weights(self):
        return self._clf.estimators_, self._clf.feature_importances_

    def load_weights(self, weights):
        self._clf.estimators_ = weights[0]
        self._clf.feature_importances_ = weights[1]

    def save_weights(self, savedir, prefix=""):
        if prefix != "":
            prefix = prefix + "_"
        with open(os.path.join(savedir, f'{prefix}estimators.npy'), 'wb') as f:
            np.save(f, self._clf.estimators_)
        with open(os.path.join(savedir, f'{prefix}feature_importances.npy'), 'wb') as f:
            np.save(f, self._clf.feature_importances_)


class DecisionTreeClassifierWrapper(SKLearnClassifierWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        self._clf = DecisionTreeClassifier(random_state=seed, **extra_clf_params)

    def set_tuned_parameters(self):
        self.tuned_parameters = {'max_depth': [3, 5, 10, 15, None]}

    def get_architecture(self):
        return self._clf.tree_.node_count, self._clf.tree_.max_depth

    def get_weights(self):
        return self._clf.tree_,

    def load_weights(self, weights):
        self._clf.tree_ = weights[0]

    def save_weights(self, savedir, prefix=""):
        with open(os.path.join(savedir, f'{prefix}tree.pkl'), 'wb') as f:
            pickle.dump(self._clf.tree_, f)

    def to_verilog(self, input_precision=4, weight_precision=8,
                   verilog_file='top.v', tb_file='top_tb.v', inputs_file='inputs.txt', output_file='output.txt', 
                   simclk_ms=1, **kwargs):
        """Creat a verilog description for the decision tree."""
        metadata = write_tree_to_verilog(
            dt_model=self._clf,
            input_bits=input_precision,
            comparator_bits=weight_precision,
            verilog_file=verilog_file,
            tb_file=tb_file,
            inputs_file=inputs_file,
            output_file=output_file,
            simclk_ms=simclk_ms,
            # rounding_f=to_fixed,
            rounding_f=lambda num, bits: np.floor(num * (2 ** bits) + 0.5).astype(np.int32),
            signed=True
        )
        if metadata is None:
            return

        input_indices = sorted(int(re.search('(\d+)', inp).group(1)) for inp in metadata['inputs'])
        metadata['exclude_indices'] = [i for i in range(self._clf.n_features_in_) if i not in input_indices]
        return metadata
    
    def fixed_point_inference(self, test_data, input_precision=4, weight_precision=8, **kwargs):
        raise NotImplementedError("Fixed point inference is not implemented for DecisionTreeClassifierWrapper.")


class DecisionTreeRegressorWrapper(SKLearnClassifierWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        self._clf = DecisionTreeRegressor(random_state=seed, **extra_clf_params)

    def set_tuned_parameters(self):
        self.tuned_parameters = {'max_depth': [3, 5, 7, 9, 11]}

    def get_architecture(self):
        return self._clf.tree_.node_count, self._clf.tree_.max_depth

    def get_weights(self):
        return self._clf.tree_,

    def load_weights(self, weights):
        self._clf.tree_ = weights[0]

    def save_weights(self, savedir, prefix=""):
        with open(os.path.join(savedir, f'{prefix}tree.pkl'), 'wb') as f:
            pickle.dump(self._clf.tree_, f)


class SVCWrapper(SKLearnClassifierWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        # self._clf = SVC(random_state=seed, **extra_clf_params)
        self._clf = LinearSVC(random_state=seed, **extra_clf_params)
        # self._clf = SGDClassifier(loss='hinge', random_state=seed, **extra_clf_params)

    def set_tuned_parameters(self):
        # Disable tuning permanently, overriding the command line argument
        self.tuned_parameters = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 50]  # only for SVC, LinearSVC
        }
        # self.tuned_parameters = {
        #     'alpha': [1e-4, 1e-3, 1e-2, 1e-1]  # only for SGDClassifier
        # }

    def get_architecture(self):
        if hasattr(self._clf, 'decision_function_shape'):
            # SVC with decision_function_shape='ovr'
            inner_function = self._clf.decision_function_shape
        elif hasattr(self._clf, 'multi_class'):
            # LinearSVC with multi_class='ovr'
            inner_function = self._clf.multi_class
        else:
            raise ValueError(f"Cannot determine the algorithm for SVM of type: {type(self._clf)}")
        return getattr(self._clf, 'n_features_in_', None), getattr(self._clf, 'classes_', None), inner_function

    def get_weights(self):
        return self._clf.coef_, self._clf.intercept_

    def load_weights(self, weights):
        self._clf.coef_ = weights[0]
        self._clf.intercept_ = weights[1]

    def save_weights(self, savedir, prefix=""):
        with open(os.path.join(savedir, f'{prefix}coef.npy'), 'wb') as f:
            np.save(f, self._clf.coef_)
        with open(os.path.join(savedir, f'{prefix}intercept.npy'), 'wb') as f:
            np.save(f, self._clf.intercept_)

    def to_verilog(self, input_precision=4, weight_precision=8,
                   verilog_file='top.v', tb_file='top_tb.v', inputs_file='inputs.txt', output_file='output.txt', 
                   simclk_ms=1, **kwargs):
        """Creat a verilog description for the SVC classifier."""
        if hasattr(self._clf, 'decision_function_shape'):
            # SVC with decision_function_shape='ovr'
            is_ovo = self._clf.decision_function_shape == 'ovo'
        elif hasattr(self._clf, 'multi_class'):
            # LinearSVC with multi_class='ovr'
            is_ovo = self._clf.multi_class == 'ovo'
        else:
            raise ValueError(f"Cannot determine the algorithm for SVM of type: {type(self._clf)}")

        metadata = write_svm_to_verilog(
            svm_model=self._clf,
            input_bits=input_precision,
            weight_bits=weight_precision,
            num_classes=len(self._clf.classes_),
            verilog_file=verilog_file,
            tb_file=tb_file,
            inputs_file=inputs_file,
            output_file=output_file,
            simclk_ms=simclk_ms,
            is_ovo=is_ovo
        )
        return metadata

    def fixed_point_inference(self, test_data, input_precision=None, weight_precision=8, **kwargs):
        """Perform fixed point inference on the SVC classifier."""
        x_test, y_test = test_data

        # from src.hw_templates.utils import svm_fxp_ps
        # from src.hw_templates.utils import convert_params_to_fxp, ConvFxp
        # inp_fxp = ConvFxp(0, 0, 8)
        # w_int_bits = min(get_width(get_maxabs(self._clf.coef_)), weight_precision - 1)
        # bias_int_bits = min(get_width(get_maxabs(self._clf.intercept_)), weight_precision - 1)
        # wfxp = ConvFxp(1, w_int_bits, weight_precision - 1 - w_int_bits)
        # bfxp = ConvFxp(1, bias_int_bits, weight_precision - 1 - bias_int_bits)
        # weights = convert_params_to_fxp(self._clf.coef_, wfxp, False)
        # biases = convert_params_to_fxp(self._clf.intercept_, bfxp, False)

        # svm_fxp = svm_fxp_ps(weights, biases, inp_fxp, wfxp, y_test)
        # return svm_fxp.get_accuracy(x_test, y_test)


        # assume that if an input precision is not given, the input is already in fixed point
        if input_precision is not None:
            x_test = convert_to_fixed_point(x_test, input_precision, fractional_bits=0)
            y_test = np.int32(y_test)

        w_int_bits = min(get_width(get_maxabs(self._clf.coef_)), weight_precision - 1)
        fxp_weights = convert_to_fixed_point(self._clf.coef_, 
                                             precision=weight_precision, 
                                             fractional_bits=weight_precision - 1 - w_int_bits, 
                                             return_type=np.int32)
        bias_int_bits = min(get_width(get_maxabs(self._clf.intercept_)), weight_precision - 1)
        fxp_intercept = convert_to_fixed_point(self._clf.intercept_, 
                                               precision=weight_precision, 
                                               fractional_bits=weight_precision - 1 - bias_int_bits, 
                                               return_type=np.int32)
        self._clf.coef_ = fxp_weights
        self._clf.intercept_ = fxp_intercept

        return self.test(x_test, y_test)


class MLPClassifierWrapper(SKLearnClassifierWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        self._clf = MLPClassifier(random_state=seed, **extra_clf_params)

    def set_tuned_parameters(self):
        self.tuned_parameters = {
            'hidden_layer_sizes': [(5,), (10,), (20,), (30,), (50,)],
        }

    def get_architecture(self):
        return [layer.shape[0] for layer in self._clf.coefs_]

    def get_weights(self):
        return self._clf.coefs_, self._clf.intercepts_

    def load_weights(self, weights):
        self._clf.coefs_ = weights[0]
        self._clf.intercepts_ = weights[1]

    def save_weights(self, savedir, prefix=""):
        for i, coef in enumerate(self._clf.coefs_):
            with open(os.path.join(savedir, f'{prefix}coefs{i}.npy'), 'wb') as f:
                np.save(f, coef)
        for i, intercept in enumerate(self._clf.intercepts_):
            with open(os.path.join(savedir, f'{prefix}intercepts{i}.npy'), 'wb') as f:
                np.save(f, intercept)
    
    def to_verilog(self, input_precision=4, weight_precision=8, 
                   verilog_file='top.v', tb_file='top_tb.v', inputs_file='inputs.txt', output_file='output.txt', 
                   simclk_ms=1, **kwargs):
        """Creat a verilog description for the MLP classifier."""
        metadata = write_mlp_to_verilog(
            mlp_model=self._clf,
            input_bits=input_precision,
            weight_bits=weight_precision,
            verilog_file=verilog_file,
            tb_file=tb_file,
            inputs_file=inputs_file,
            output_file=output_file,
            simclk_ms=simclk_ms
        )
        return metadata

    def fixed_point_inference(self, test_data, input_precision=None, weight_precision=8, **kwargs):
        """Perform fixed point inference on the SVC classifier."""
        x_test, y_test = test_data

        from src.hw_templates.utils import mlp_fxp_ps
        from src.hw_templates.utils import convertCoef, convertIntercepts, ConvFxp

        inp_fxp = ConvFxp(0, 0, input_precision if input_precision is not None else 8)
        w_int_bits = min(get_width(get_maxabs(self._clf.coefs_[0])), weight_precision - 1)
        bias_int_bits = min(get_width(get_maxabs(self._clf.intercepts_[0])), weight_precision - 1)
        wfxp = ConvFxp(1, w_int_bits, weight_precision - 1 - w_int_bits)
        bfxp=[]
        trlst=[0]
        for i,c in enumerate(self._clf.intercepts_):
            trlst.append(0)
            b_int=get_width(get_maxabs(c))
            bfxp.append(ConvFxp(1,b_int,inp_fxp.frac+(i+1)*wfxp.frac-trlst[i]))
        intercepts=convertIntercepts(self._clf.intercepts_, bfxp, False)
        coefficients=convertCoef(self._clf.coefs_, wfxp, False)

        mlp_fxp = mlp_fxp_ps(coefficients, intercepts, inp_fxp, wfxp,  0, y_test, self._clf)
        fxp_accuracy = mlp_fxp.get_accuracy(x_test, y_test)
        print(f"First accuracy: {fxp_accuracy}")

        return fxp_accuracy


class BNNWrapper(SKLearnClassifierWrapper):
    def set_clf(self, *args, **extra_clf_params):
        num_classes = extra_clf_params.get('num_classes')
        num_features = extra_clf_params.get('num_features')
        input_bitwidth = extra_clf_params.get('input_bitwidth', 8)
        # num_nodes = extra_clf_params.get('num_nodes', 10)
        # learning_rate = extra_clf_params.get('learning_rate', 0.01)
        build_fn = partial(self.define_model,
                           num_classes=num_classes,
                           input_bitwidth=input_bitwidth,
                           num_features=num_features)
        self._clf = KerasClassifier(build_fn=build_fn)

        self.num_classes = num_classes
        self.num_features = num_features
        
    def set_tuned_parameters(self):
        self.tuned_parameters = {
            'num_nodes': [(10,), (20,), (50,), (100,)],
            'learning_rate': [0.001, 0.01, 0.1],
        }

    @staticmethod
    def define_model(num_classes, input_bitwidth, num_features, num_nodes=20, learning_rate=0.01):
        model = Sequential()
        model.add(keras.Input(shape=(num_features,)))
        model.add(QActivation(f"quantized_bits({input_bitwidth}, keep_negative=0)"))
        model.add(QDense(units=num_nodes,
                         kernel_quantizer="ternary",
                         use_bias=False,))
                        #  input_shape=(num_features,)))
        model.add(QActivation("binary"))
        model.add(QDense(units=num_classes,
                         kernel_quantizer='ternary',
                         use_bias=False))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            # loss='sparse_categorical_crossentropy',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class TNNWrapper(SKLearnClassifierWrapper):
    def set_clf(self, *args, **extra_clf_params):
        num_classes = extra_clf_params.get('num_classes')
        num_features = extra_clf_params.get('num_features')
        input_bitwidth = extra_clf_params.get('input_bitwidth', 8)
        # num_nodes = extra_clf_params.get('num_nodes', 10)
        # learning_rate = extra_clf_params.get('learning_rate', 0.01)
        build_fn = partial(self.define_model,
                           num_classes=num_classes,
                           input_bitwidth=input_bitwidth,
                           num_features=num_features)
        self._clf = KerasClassifier(build_fn=build_fn)

        self.num_classes = num_classes
        self.num_features = num_features
        
    def set_tuned_parameters(self):
        self.tuned_parameters = {
            'num_nodes': [(10,), (20,), (50,), (100,)],
            'learning_rate': [0.001, 0.01, 0.1],
        }

    @staticmethod
    def define_model(num_classes, input_bitwidth, num_features, num_nodes=20, learning_rate=0.01):
        model = Sequential()
        model.add(keras.Input(shape=(num_features,)))
        model.add(QActivation(f"quantized_bits({input_bitwidth}, keep_negative=0)"))
        model.add(QDense(units=num_nodes,
                         kernel_quantizer="binary",
                         use_bias=False,))
                        #  input_shape=(num_features,)))
        model.add(QActivation("binary"))
        model.add(QDense(units=num_classes,
                         kernel_quantizer='binary',
                         use_bias=False))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            # loss='sparse_categorical_crossentropy',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class KerasNNWrapper(_ClassifierWrapper):
    def set_clf(self, seed=None, **extra_params):
        num_classes = extra_params.get('num_classes')
        num_features = extra_params.get('num_features')
        input_bitwidth = extra_params.get('input_bitwidth', 8)
        num_nodes = extra_params.get('num_nodes', 10)
        learning_rate = extra_params.get('learning_rate', 0.01)
        model = self.define_model(num_classes, input_bitwidth, num_features, num_nodes, learning_rate)
        self._clf = model
        self.num_classes = num_classes
        self.num_features = num_features

    def set_accuracy_function(self, accuracy_metric):
        self.accuracy_metric = accuracy_metric
        self.accuracy_params = {}
        if accuracy_metric == AccuracyMetric.Accuracy:
            self.accuracy_f = accuracy_keras
        elif accuracy_metric == AccuracyMetric.F1:
            self.accuracy_f = f1score_keras
            self.accuracy_params['average'] = 'weighted'
        else:
            raise ValueError(f"Unknown accuracy metric: {accuracy_metric}")

    def get_tuned_parameters(self, hp, **extra_params):
        num_nodes = hp.Int("hidden_neurons", min_value=10, max_value=100, step=10)
        learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.01)
        num_classes = extra_params.get('num_classes')
        input_bitwidth = extra_params.get('input_bitwidth', 8)
        num_features = extra_params.get('num_features')
        return num_classes, input_bitwidth, num_features, num_nodes, learning_rate

    def tune(self, train_data, **extra_clf_params):
        def my_builder(hp):
            params = self.get_tuned_parameters(hp, **extra_clf_params)
            return self.define_model(*params)

        x_train, y_train = train_data
        x_test, y_test = extra_clf_params.pop('test_data', (None, None))
        y_train = transform_categorical(y_train, self.num_classes)
        y_test = transform_categorical(y_test, self.num_classes)

        logdir = extra_clf_params.pop('logdir', getattr(logging.getLogger(), 'logdir', None))
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=my_builder,
            objective="val_accuracy",
            max_trials=20,
            directory=os.path.join(logdir, 'bo_tuner') if logdir else None,
            overwrite=True,
            # project_name=os.path.join(logdir, 'bo_tuner') if logdir else None,
            # logger=logger,
        )
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5,
                                                   restore_best_weights=True,
                                                   verbose=0)
        # search for the model with the best test accuracy 
        tuner.search(x_train, y_train,
                     epochs=50,
                     validation_data=(x_test, y_test),
                     verbose=0,
                     callbacks=[early_stop])
        #tuna.results_summary()
        model = tuner.get_best_models(num_models=1)[0]

        # get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        for key, value in best_hp.values.items():
            logger.debug(f"Best {key.replace('_', ' ')}: {value}")

        return model
    
    def train(self, x_train, y_train, x_test=None, y_test=None, **train_kwargs):
        y_train = transform_categorical(y_train, self.num_classes)
        y_test = transform_categorical(y_test, self.num_classes)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5,
                                                   restore_best_weights=True,
                                                   verbose=0)
        train_kwargs['callbacks'] = train_kwargs.get('callbacks', []) + [early_stop]
        train_kwargs['epochs'] = train_kwargs.get('epochs', 50)
        train_kwargs['verbose'] = train_kwargs.get('verbose', 0)
        train_kwargs['validation_data'] = (x_test, y_test)
        return super().train(x_train, y_train, x_test, y_test, **train_kwargs)

    def test(self, x_test, y_test):
        y_pred = self._clf.predict(x_test, verbose=0)
        if isinstance(y_test, (tuple, list)):
            y_test = transform_categorical(y_test, self.num_classes)
        accuracy = self.accuracy_f(y_test, y_pred, **self.accuracy_params)
        return accuracy

    def get_weights(self):
        return self._clf.get_weights()

    def load_weights(self, weights):
        self._clf.set_weights(weights)

    def save_weights(self, savedir, prefix=""):
        if prefix == "":
            prefix = self.__class__.__name__
        elif prefix.endswith('.'):
            prefix = prefix.replace('.', '')
        savefile = os.path.join(savedir, f'{prefix}.weights.h5')
        self._clf.save_weights(savefile)

        savefile = os.path.join(savedir, f'{prefix}.quant.weights.h5')
        qweights = qutils.model_save_quantized_weights(self._clf, savefile)

        # NOTE: Weights can be saved via 'qweights', which is a dictionary with
        #       keys as layer names and values as quantized weights. For BNN/TNN,
        #       there seem to be multiple weight values per layer (more than 2/3),
        #       so we save them using the np.sign() function. That way, we get
        #       an array of [-1, 1] for BNNs and [-1, 0, 1] for TNNs.

        # save quantized weights as numpy arrays
        dummy_clf = deepcopy(self._clf)
        dummy_clf.load_weights(savefile)
        quantized_weights = dummy_clf.get_weights()
        for layer_id, q_layer_w in enumerate(quantized_weights):
            np.save(os.path.join(savedir, f'{prefix}.qweights.layer{layer_id}.npy'), q_layer_w)


class TNNKerasWrapper(KerasNNWrapper):
    @staticmethod
    def define_model(num_classes, input_bitwidth, num_features, num_nodes, learning_rate):
        model = Sequential()
        model.add(keras.Input(shape=(num_features,)))
        model.add(QActivation(f"quantized_bits({input_bitwidth}, keep_negative=0)"))
        model.add(QDense(units=num_nodes,
                         kernel_quantizer="ternary",
                         use_bias=False,))
                        #  input_shape=(num_features,)))
        model.add(QActivation("binary"))
        model.add(QDense(units=num_classes,
                         kernel_quantizer='ternary',
                         use_bias=False))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            # loss='sparse_categorical_crossentropy',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class BNNKerasWrapper(KerasNNWrapper):
    @staticmethod
    def define_model(num_classes, input_bitwidth, num_features, num_nodes, learning_rate):
        model = Sequential()
        model.add(keras.Input(shape=(num_features,)))
        model.add(QActivation(f"quantized_bits({input_bitwidth}, keep_negative=0)"))
        model.add(QDense(units=num_nodes,
                         kernel_quantizer="binary",
                         use_bias=False,))
                        #  input_shape=(num_features,)))
        model.add(QActivation("binary"))
        model.add(QDense(units=num_classes,
                         kernel_quantizer='binary(alpha=1)',
                         use_bias=False))
        model.add(layers.Activation('softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            # loss='sparse_categorical_crossentropy',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class FCNNKerasWrapper(KerasNNWrapper):
    def set_clf(self, seed=None, **extra_clf_params):
        num_classes = extra_clf_params.get('num_classes')
        num_features = extra_clf_params.get('num_features')
        input_bitwidth = extra_clf_params.get('input_bitwidth', 8)
        num_nodes = extra_clf_params.get('num_nodes', 10)
        learning_rate = extra_clf_params.get('learning_rate', 0.01)
        feature_costs = extra_clf_params.get('feature_costs', [])
        lambda_reg = extra_clf_params.get('lambda_reg', 0.001)

        model = self.define_model(num_classes, input_bitwidth, num_features, num_nodes, 
                                  learning_rate, feature_costs, lambda_reg)
        self._clf = model
        self.num_classes = num_classes
        self.num_features = num_features
        self.feature_costs = feature_costs

    @staticmethod
    def define_model(num_classes, input_bitwidth, num_features, num_nodes=20, learning_rate=0.01,
                     feature_costs=None, lambda_reg=0.01):
        model = Sequential()

        # model.add(keras.Input(shape=(num_features,)))
        model.add(InputFeatureGates(num_features=num_features, feature_costs=feature_costs, lambda_reg=lambda_reg))

        model.add(layers.Dense(units=num_nodes,
                               activation='relu',
                               input_shape=(num_features,)))
        model.add(layers.Dense(units=num_classes,
                               activation='softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
