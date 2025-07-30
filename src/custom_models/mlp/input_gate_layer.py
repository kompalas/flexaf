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


class ConcreteGate(tf.keras.layers.Layer):
    def __init__(self,
                 num_features,
                 feature_costs,
                 lambda_reg=0.1,
                 temperature=0.1,
                 warmup_epochs=5,
                 init_mean=0.0,
                 init_std=1.0,
                 epsilon=1e-7,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_costs = tf.constant(feature_costs, dtype=tf.float32)
        # Regularization strength (L0 weight)
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.epsilon = epsilon
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32, name='epoch_var')

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomNormal(mean=self.init_mean, stddev=self.init_std)
        self.log_alpha = self.add_weight(
            name="log_alpha",
            shape=(self.num_features,),
            initializer=initializer,
            trainable=True
        )

    def call(self, inputs, training=True):
        if training:
            u = tf.random.uniform(shape=(self.num_features,), minval=0.0, maxval=1.0)
            s = tf.sigmoid((tf.math.log(u + self.epsilon) - tf.math.log(1 - u + self.epsilon) + self.log_alpha) / self.temperature)
            z = tf.clip_by_value(s, 0.0, 1.0)
        else:
            z = tf.sigmoid(self.log_alpha)

        z = tf.cond(
            self.epoch_var < self.warmup_epochs,
            lambda: tf.stop_gradient(z),
            lambda: z
        )

        # L0-inspired regularization: Penalize expected gate usage
        gate_probs = tf.sigmoid(self.log_alpha)  # expected "on" prob for each feature
        expected_cost = tf.reduce_sum(gate_probs * self.feature_costs)  # cost-weighted sum
        self.add_loss(self.lambda_reg * expected_cost)  # apply as auxiliary loss

        return inputs * z

    def on_epoch_end(self):
        self.epoch_var.assign_add(1)

    def get_gate_values(self):
        return tf.sigmoid(self.log_alpha).numpy()
