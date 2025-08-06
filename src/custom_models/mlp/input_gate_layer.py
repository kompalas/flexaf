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


class CustomBinaryInitializer(tf.keras.initializers.Initializer):
    def __init__(self, binary_mask):
        self.binary_mask = tf.convert_to_tensor(binary_mask, dtype=tf.float32)

    def __call__(self, shape, dtype=None):
        return tf.where(self.binary_mask > 0.5, 10.0, -10.0)  # log_alpha ≈ ±10


class ConcreteGate(tf.keras.layers.Layer):
    def __init__(self,
                 num_features,
                 feature_costs,
                 lambda_reg=0.1,
                 temperature=0.1,
                 warmup_epochs=5,
                 init_mean=0.0,
                 init_std=1.0,
                 initial_binary_mask=None,
                 epsilon=1e-7,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_costs = tf.constant(feature_costs, dtype=tf.float32)
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.epsilon = epsilon
        self.binary_mask = initial_binary_mask
        self.epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32, name='epoch_var')

    def build(self, input_shape):
        if self.binary_mask is not None:
            assert len(self.binary_mask) == self.num_features, \
                "Length of binary_mask must match num_features"
            initializer = CustomBinaryInitializer(self.binary_mask)
        else:
            initializer = tf.keras.initializers.RandomNormal(mean=self.init_mean, stddev=self.init_std)

        self.log_alpha = self.add_weight(
            name="log_alpha",
            shape=(self.num_features,),
            initializer=initializer,
            trainable=True
        )

    def call(self, inputs, training=True):
        deterministic_gate = tf.sigmoid(self.log_alpha)

        if training:
            u = tf.random.uniform(shape=(self.num_features,), minval=0.0, maxval=1.0)
            s = tf.sigmoid((tf.math.log(u + self.epsilon) - tf.math.log(1 - u + self.epsilon) + self.log_alpha) / self.temperature)
            z_sampled = tf.clip_by_value(s, 0.0, 1.0)

            z = tf.cond(
                self.epoch_var < self.warmup_epochs,
                lambda: tf.stop_gradient(deterministic_gate),  # constant gate
                lambda: z_sampled
            )
        else:
            z = deterministic_gate  # use expected value at inference

        # Regularization loss (expected cost of active gates)
        gate_probs = tf.sigmoid(self.log_alpha)
        expected_cost = tf.reduce_sum(gate_probs * self.feature_costs)
        self.add_loss(self.lambda_reg * expected_cost)

        return inputs * z

    def on_epoch_end(self):
        self.epoch_var.assign_add(1)

    def get_gate_values(self):
        return tf.sigmoid(self.log_alpha).numpy()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "feature_costs": self.feature_costs.numpy().tolist(),
            "lambda_reg": self.lambda_reg,
            "temperature": self.temperature,
            "warmup_epochs": self.warmup_epochs,
            "init_mean": self.init_mean,
            "init_std": self.init_std,
            "initial_binary_mask": self.binary_mask,
            "epsilon": self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Convert list back to tensor
        config["feature_costs"] = tf.constant(config["feature_costs"], dtype=tf.float32)
        return cls(**config)