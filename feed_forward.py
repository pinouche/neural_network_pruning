import tensorflow as tf
import keras
import pickle


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, epoch_list):
        self.epoch_list = epoch_list

    def on_epoch_end(self, epoch, logs={}):
        if epoch+1 in self.epoch_list:
            weights_trained = self.model.get_weights()
            pickle.dump(weights_trained, open("trained_weights_" + str(epoch+1) + ".pickle", 'wb'))


class Selu(keras.layers.Layer):
    def __init__(self, units, input_dim, seed, weights=None, bias=None, mask=None):
        super(Selu, self).__init__()

        tf.set_random_seed(seed)
        if mask is None:
            mask = 1

        if weights is None:
            w_init = tf.glorot_normal_initializer()
            self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True)

            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype="float32"), trainable=True)

        else:
            self.w = tf.Variable(initial_value=weights, trainable=True, dtype=tf.float32)
            if mask is not None:
                self.w = tf.cast(self.w, tf.float32) * mask

            # do not prune the weights
            self.b = tf.Variable(initial_value=bias, trainable=True, dtype=tf.float32)

    def call(self, inputs):

        activation_layer = keras.activations.selu(tf.matmul(inputs, self.w) + self.b)

        return activation_layer


class Output(keras.layers.Layer):
    def __init__(self, output_dim, input_dim, seed, weights=None, mask=None):
        super(Output, self).__init__()

        tf.set_random_seed(seed)
        if mask is None:
            mask = 1

        if weights is None:
            w_init = tf.glorot_normal_initializer()
            self.w = tf.Variable(initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"), trainable=True)

        else:
            self.w = tf.Variable(initial_value=weights, trainable=True, dtype=tf.float32)
            if mask is not None:
                self.w = self.w * mask

    def call(self, inputs):

        activation_layer = tf.matmul(inputs, self.w)

        return activation_layer


def init_keras_variables_session():

    keras.backend.set_session(tf.compat.v1.Session())
    init = tf.global_variables_initializer()
    keras.backend.get_session().run(init)


def feed_forward_model(hidden_size_list, input_size, output_size, seed, weights_list, mask_list):

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_size,)))

    weight_bias_index = 0
    for index in range(len(hidden_size_list)):
        if index == 0:
            # input layer
            model.add(Selu(hidden_size_list[index], input_size, seed, weights_list[weight_bias_index],
                           weights_list[weight_bias_index + 1], mask_list[weight_bias_index]))

        else:
            model.add(Selu(hidden_size_list[index], hidden_size_list[index - 1], seed, weights_list[weight_bias_index],
                           weights_list[weight_bias_index + 1], mask_list[weight_bias_index]))
        weight_bias_index += 2

    # output layer
    model.add(Output(output_size, hidden_size_list[-1], seed, weights_list[-1]))
    model.add(keras.layers.Activation(keras.activations.softmax))

    return model


def model_keras(seed, data, hidden_size_list=None, weights_list=None, mask_list=None):
    if data == "mnist":
        number_of_layers = 1
        hidden_size = 512
        input_size = 784
        output_size = 10

        if hidden_size_list is None:
            hidden_size_list = [hidden_size] * number_of_layers
        if weights_list is None:
            weights_list = [None] * int(number_of_layers*2+1)
        if mask_list is None:
            mask_list = [None] * int(number_of_layers*2+1)

        model = feed_forward_model(hidden_size_list, input_size, output_size, seed, weights_list, mask_list)

    elif data == "cifar10":
        number_of_layers = 3
        hidden_size = 100
        input_size = 3072
        output_size = 10

        if hidden_size_list is None:
            hidden_size_list = [hidden_size] * number_of_layers
        if weights_list is None:
            weights_list = [None] * int(number_of_layers*2+1)  # weight, bias per layer and only weight for output layer
        if mask_list is None:
            mask_list = [None] * int(number_of_layers*2+1)

        model = feed_forward_model(hidden_size_list, input_size, output_size, seed, weights_list, mask_list)

    else:
        raise Exception("wrong dataset")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model

