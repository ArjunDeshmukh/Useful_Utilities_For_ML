import tensorflow as tf
from tensorflow.keras import backend
import tempfile
import sys
import os

__version__ = 'dev'

class L2_SP(tf.keras.regularizers.Regularizer):
    def __init__(self, base_weights, l2=0.01):
        '''
            base_weights has to be a numpy array
        '''

        l2 = 0.01 if l2 is None else l2
        keras.regularizers._check_penalty_number(l2)
        self.l2 = backend.cast_to_floatx(l2)

        self.base_weights_np_arr = base_weights

        self.base_weights_tensor = tf.convert_to_tensor(base_weights)

    def __call__(self, x):
        return 2.0 * self.l2 * tf.nn.l2_loss(tf.math.subtract(x, self.base_weights_tensor))

    def get_config(self):
        return {"l2": float(self.l2), "base_weights": self.base_weights_np_arr}


def add_kernel_regularization(model, regularization_weight=0.0001, regularization_type='L2'):
    for layer in model.layers:
        attr = 'kernel_regularizer'
        if hasattr(layer, 'kernel_regularizer'):
            if regularization_type == 'L1':
                regularizer = tf.keras.regularizers.L1(regularization_weight)
                setattr(layer, attr, regularizer)
            elif regularization_type == 'L2':
                regularizer = tf.keras.regularizers.L2(regularization_weight)
                setattr(layer, attr, regularizer)
            elif regularization_type == 'L2_SP':
                base_weights = layer.weights[0].numpy()
                regularizer = L2_SP(base_weights=base_weights, l2=regularization_weight)
                setattr(layer, attr, regularizer)
            else:
                sys.stderr.write('The regularization_type variable should be one of these: L2, L1, L2_SP')

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json, custom_objects={'L2_SP': L2_SP})

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model
