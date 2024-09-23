import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Custom Dropout Layer
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, initial_rate=0.5, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = initial_rate  # Initial dropout rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

    def set_rate(self, new_rate):
        self.rate = new_rate  # Dynamically update dropout rate

    def get_config(self):
        # Return the configuration of this layer for serialization
        config = super().get_config()
        config.update({
            'initial_rate': self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Custom L2 Regularizer
class CustomL2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, initial_l2=0.01):
        self.l2 = initial_l2

    def __call__(self, x):
        return self.l2 * tf.reduce_sum(tf.square(x))

    def set_l2(self, new_l2):
        self.l2 = new_l2

    def get_config(self):
        # Return a dictionary that describes the configuration of this regularizer
        return {'l2': self.l2}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PID_L2_DROPOUT(Callback):
    def __init__(self, K_p=0.1, K_i=0.01, K_d=0.05, start_epoch=5, clamp_range=0.05):
        super(PID_L2_DROPOUT, self).__init__()
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.start_epoch = start_epoch
        self.cumulative_error = 0
        self.previous_error = 0
        self.clamp_range = clamp_range
        self.modified_layers = False  # Flag to track if we already modified layers

    def pid_adjustment(self, current_error):
        # Proportional term
        P = self.K_p * current_error
        # Integral term
        self.cumulative_error += current_error
        I = self.K_i * self.cumulative_error
        # Derivative term
        D = self.K_d * (current_error - self.previous_error)
        self.previous_error = current_error
        adjustment = P + I + D
        return max(-self.clamp_range, min(self.clamp_range, adjustment))

    def adjust_regularization(self, adjustment):
        """Adjust dropout and L2 regularization dynamically based on PID output."""
        for layer in self.model.layers:
            if isinstance(layer, CustomDropout):
                new_rate = min(0.5, max(0.0, layer.rate + adjustment))
                layer.set_rate(new_rate)
                print(f"Adjusted Dropout Rate to: {new_rate:.6f}")

            if hasattr(layer, 'kernel_regularizer') and isinstance(layer.kernel_regularizer, CustomL2Regularizer):
                current_l2_value = layer.kernel_regularizer.l2
                new_l2 = min(0.1, max(0.00005, current_l2_value + adjustment))
                layer.kernel_regularizer.set_l2(new_l2)
                print(f"Adjusted L2 Regularization to: {new_l2:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            train_loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            error = val_loss - train_loss
            adjustment = self.pid_adjustment(error)
            self.adjust_regularization(adjustment)
  
def change_model(model):
    """Rebuild the model by replacing Dropout and L2 regularizers."""
    new_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            # Replace Dropout layers with CustomDropout, keeping the current dropout rate
            new_dropout_layer = CustomDropout(initial_rate=layer.rate)
            new_layers.append(new_dropout_layer)
            print(f"Replaced Dropout layer with CustomDropout with rate: {layer.rate}")
        elif hasattr(layer, 'kernel_regularizer') and isinstance(layer.kernel_regularizer, tf.keras.regularizers.L2):
            # Update L2 regularizers without replacing the layer
            current_l2_value = layer.kernel_regularizer.l2  # Get the current L2 regularization value
            new_l2_regularizer = CustomL2Regularizer(initial_l2=current_l2_value)
            # Clone the layer but replace the regularizer
            config = layer.get_config()
            config['kernel_regularizer'] = new_l2_regularizer
            new_layer = layer.__class__.from_config(config)
            new_layers.append(new_layer)
            print(f"Updated L2 Regularizer in layer with L2 value: {current_l2_value}")
        else:
            # Keep the layer as-is if no replacement is needed
            new_layers.append(layer)

    # Rebuild the model using the new layers
    new_model = tf.keras.Sequential(new_layers)

    # Copy weights from the old model to the new one
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        new_layer.set_weights(old_layer.get_weights())

    model = new_model
    return model