from keras.layers import Layer
import tensorflow as tf


class InputFeatureGates(Layer):
    def __init__(self, num_features, feature_costs, lambda_reg=0.1):
        super().__init__()
        self.num_features = num_features
        self.feature_costs = tf.constant(feature_costs, dtype=tf.float32)
        self.lambda_reg = lambda_reg

    def build(self, input_shape):
        self.gates = self.add_weight(
            shape=(self.num_features,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        )
        # Add regularization loss
        self.add_loss(self.lambda_reg * tf.reduce_sum(self.gates * self.feature_costs))

    def call(self, inputs):
        return inputs * self.gates  # Element-wise multiply
