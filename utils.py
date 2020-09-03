import numpy as np
import pickle
import keras


# load data functions
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    return x_train, x_test, y_train, y_test


def load_cifar(flatten=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], 3072))
        x_test = np.reshape(x_test, (x_test.shape[0], 3072))

    return x_train, x_test, y_train, y_test


def keras_get_gradient_weights(model, data_x):
    batch_size = 512
    output = model.layers[-2].output
    trainable_weights_list = model.trainable_weights
    gradients = keras.backend.gradients(output, trainable_weights_list)
    get_gradients = keras.backend.function(model.inputs[0], gradients)

    n, p = data_x.shape[0], data_x.shape[1]

    gradient_list = []
    for index in range(batch_size):
        gradient_list.append(get_gradients([data_x[index, :].reshape(1, p)]))

    gradient_list = np.mean(np.abs(np.array(gradient_list)), axis=0)

    return gradient_list


def keras_get_hidden_layers(model, data_x):
    def keras_function_layer(model_layer, data):
        hidden_func = keras.backend.function(model.layers[0].input, model_layer.output)
        result = hidden_func([data])

        return result

    hidden_layers_list = []
    for index in range(len(model.layers) - 2):
        hidden_layer = keras_function_layer(model.layers[index], data_x)
        hidden_layers_list.append(hidden_layer)

    return hidden_layers_list


def get_gradients_hidden_layers(model, data_x):
    hidden_layers_list = [layer.output for layer in model.layers[:-2]]
    inputs = model.inputs[0]
    output = model.layers[-2].output

    gradients = keras.backend.gradients(output, hidden_layers_list)
    get_gradients = keras.backend.function(inputs, gradients)

    p = data_x.shape[1]

    gradient_list = []
    for index in range(data_x.shape[0]):
        gradient_list.append(get_gradients([data_x[index, :].reshape(1, p)]))

    return gradient_list


def get_corr(hidden_representation_list_one):
    list_corr_matrices = []

    for index in range(len(hidden_representation_list_one)):
        hidden_representation_one = hidden_representation_list_one[index]

        n = hidden_representation_one.shape[1]
        corr_matrix_nn = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                corr = np.corrcoef(hidden_representation_one[:, i], hidden_representation_one[:, j])[0, 1]
                corr_matrix_nn[i, j] = corr

        list_corr_matrices.append(corr_matrix_nn)

    return list_corr_matrices


def add_noise(parent_weights, seed, t=0.5):
    np.random.seed(seed)

    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))
    mean_parent, std_parent = 0.0, np.std(parent_weights)

    weight_noise = np.random.normal(loc=mean_parent, scale=std_parent, size=parent_weights.shape)
    parent_weights = (t * parent_weights + (1 - t) * weight_noise) * scale_factor

    return parent_weights






