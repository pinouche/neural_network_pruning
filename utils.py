import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os
from tensorflow.python import keras


# load data functions
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_mnist():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    return x_train, x_test, y_train, y_test


def load_cifar():
    path = "../../Documents/cifar-10-python/cifar-10-batches-py"

    list_y = []
    list_x = []
    for file in os.listdir(path):
        if "data" in file:
            full_path = path + "/" + file
            dict_batch = unpickle(full_path)
            list_y.append(dict_batch[b'labels'])
            list_x.append(dict_batch[b'data'])

    y_train = np.asarray([val for sublist in list_y for val in sublist])
    x_train = np.asarray([val for sublist in list_x for val in sublist])

    test_path = path + "/" + "test_batch"
    dict_test = unpickle(test_path)
    y_test, x_test = np.asarray(dict_test[b'labels']), np.asarray(dict_test[b'data'])

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, x_test, y_train, y_test


def selu(x, lamb=1.0507, alpha=1.67326):
    x[x > 0] = x[x > 0] * lamb
    x[x <= 0] = lamb * (alpha * np.exp(x[x <= 0]) - alpha)

    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


def one_hot(array):
    enc = OneHotEncoder(categories='auto')
    array = np.reshape(array, (array.shape[0], 1))
    enc.fit(array)
    array_hot = enc.transform(array)

    return array_hot


def get_intermediate_layers(data_x, list_weights):
    num_hidden_layers = int((len(list_weights) - 1) / 2)
    list_hidden_representation = []
    count = 0
    for layer in range(num_hidden_layers):

        if layer == 0:
            hidden_layer = np.matmul(data_x, list_weights[count]) + list_weights[count + 1]
            hidden_layer = selu(hidden_layer)
        else:
            hidden_layer = np.matmul(hidden_layer, list_weights[count]) + list_weights[count + 1]
            hidden_layer = selu(hidden_layer)
        list_hidden_representation.append(hidden_layer)
        count += 2

    return list_hidden_representation


# cross correlation function for both bipartite matching (hungarian method)
def get_corr(relu_original_one, relu_original_two, crossover="unsafe"):
    axis_number = 0
    semi_matching = False
    n = relu_original_one.shape[1]

    scaler = StandardScaler()  # Fit your data on the scaler object
    relu_layer_one = scaler.fit_transform(relu_original_one)
    relu_layer_two = scaler.fit_transform(relu_original_two)

    # get the correlation matrix for the neurons
    corr_matrix_nn = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(relu_layer_one[:, i], relu_layer_two[:, j])[0, 1]
            corr_matrix_nn[i, j] = corr

    if crossover == "unsafe":
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "safe":
        corr_matrix_nn *= -1  # default of linear_sum_assignement is to minimize cost, we want to max "cost"
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "orthogonal":
        corr_matrix_nn = np.abs(corr_matrix_nn)
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "normed":
        corr_matrix_nn = np.abs(corr_matrix_nn)
        corr_matrix_nn *= -1
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "naive":
        list_neurons_x, list_neurons_y = list(range(corr_matrix_nn.shape[0])), list(range(corr_matrix_nn.shape[0]))
    else:
        raise ValueError('the crossover method is not defined')

    return corr_matrix_nn, list_neurons_x, list_neurons_y


def get_network_similarity(list_corr_matrices, list_ordered_indices_one, list_ordered_indices_two):
    list_meta = []

    for layer_num in range(len(list_corr_matrices)):
        list_corr = []
        for index in range(len(list_ordered_indices_one)):
            i = list_ordered_indices_one[layer_num][index]
            j = list_ordered_indices_two[layer_num][index]
            corr = list_corr_matrices[layer_num][i][j]
            list_corr.append(corr)

        list_meta.append(np.mean(list_corr))

    similarity = np.mean(list_meta)

    return similarity


# Algorithm 2
def order_weights(nn_weights_list, list_indices_hidden):
    count = 0
    depth = count * 2
    for layer in range(len(list_indices_hidden)):
        for index in range(3):
            if index == 0:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][:,
                                                 list_indices_hidden[layer]]  # order columns for weights
            elif index == 1:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][
                    list_indices_hidden[layer]]  # order columns for bias
            elif index == 2:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][list_indices_hidden[layer],
                                                 :]  # order rows

        count += 1
        depth = count * 2

    nn_weights_list = np.asarray(nn_weights_list)

    return nn_weights_list


def crossover_method(x_data, weights_nn_one, weights_nn_two, crossover):
    my_range = list(np.arange(-0.5, 1.52, 1 / 50))
    my_range[25] = 0
    my_range[50] = 0.5

    list_hidden_representation_one = get_intermediate_layers(x_data, weights_nn_one)
    list_hidden_representation_two = get_intermediate_layers(x_data, weights_nn_two)

    list_ordered_indices_one = []
    list_ordered_indices_two = []
    list_corr_matrices = []
    for index in range(len(list_hidden_representation_one)):
        hidden_layer_one = list_hidden_representation_one[index]
        hidden_layer_two = list_hidden_representation_two[index]

        corr_matrix_nn, indices_one, indices_two = get_corr(hidden_layer_one, hidden_layer_two, crossover)
        list_corr_matrices.append(corr_matrix_nn)
        list_ordered_indices_one.append(indices_one)
        list_ordered_indices_two.append(indices_two)

    # order the weight matrices

    if crossover == "naive":
        list_ordered_weights_one = weights_nn_one[:]
        list_ordered_weights_two = weights_nn_two[:]

    else:
        weights_nn_one_copy = weights_nn_one[:]
        weights_nn_two_copy = weights_nn_two[:]
        list_ordered_weights_one = order_weights(weights_nn_one_copy, list_ordered_indices_one)
        list_ordered_weights_two = order_weights(weights_nn_two_copy, list_ordered_indices_two)

    return list_ordered_weights_one, list_ordered_weights_two


def arithmetic_crossover(network_one, network_two, index):

    t = 0.5
    scale_factor = np.sqrt(1/(np.power(t, 2)+np.power(1-t, 2)))

    list_weights = []
    for index in range(len(network_one)):
        averaged_weights = (t*network_one[index] + (1-t)*network_two[index])*scale_factor
        list_weights.append(averaged_weights)

    return list_weights






