import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
import pickle
import os
from tensorflow.python import keras


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


def selu(x, lamb=1.0507, alpha=1.67326):
    x[x > 0] = x[x > 0] * lamb
    x[x <= 0] = lamb * (alpha * np.exp(x[x <= 0]) - alpha)

    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


def add_noise(parent_weights, t, seed):
    np.random.seed(seed)

    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))
    mean_parent, std_parent = 0.0, np.std(parent_weights)
    weight_noise = np.random.normal(loc=mean_parent, scale=std_parent, size=parent_weights.shape)
    parent_weights = (t * parent_weights + (1 - t) * weight_noise) * scale_factor

    return parent_weights


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


def get_corr(hidden_representation_list_one, hidden_representation_list_two):
    list_corr_matrices = []

    for index in range(len(hidden_representation_list_one)):
        hidden_representation_one = hidden_representation_list_one[index]
        hidden_representation_two = hidden_representation_list_two[index]

        n = hidden_representation_one.shape[1]
        scaler = StandardScaler()  # Fit your data on the scaler object
        hidden_representation_one = scaler.fit_transform(hidden_representation_one)
        hidden_representation_two = scaler.fit_transform(hidden_representation_two)

        corr_matrix_nn = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                corr = np.corrcoef(hidden_representation_one[:, i], hidden_representation_two[:, j])[0, 1]
                corr_matrix_nn[i, j] = corr

        list_corr_matrices.append(corr_matrix_nn)

    return list_corr_matrices


# cross correlation function for both bipartite matching (hungarian method)
def bipartite_matching(corr_matrix_nn, crossover="unsafe"):
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

    return list_neurons_x, list_neurons_y


def get_network_similarity(list_corr_matrices, list_ordered_indices_one, list_ordered_indices_two):
    list_meta = []

    for layer_num in range(len(list_corr_matrices)):
        list_corr = []
        for index in range(len(list_ordered_indices_one)):
            i = list_ordered_indices_one[layer_num][index]
            j = list_ordered_indices_two[layer_num][index]
            corr = np.abs(list_corr_matrices[layer_num][i][j])
            list_corr.append(corr)

        list_meta.append(np.mean(list_corr))

    similarity = np.mean(list_meta)

    return similarity


# Algorithm 2
def apply_mask_to_weights(nn_weights_list, list_indices_hidden):
    count = 0
    depth = count * 2
    for layer in range(len(list_indices_hidden)):
        for index in range(3):
            if index == 0:
                # order columns for weights
                nn_weights_list[index + depth] = nn_weights_list[index + depth][:, list_indices_hidden[layer]]
            elif index == 1:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][
                    list_indices_hidden[layer]]  # order columns for bias
            elif index == 2:
                # order rows
                nn_weights_list[index + depth] = nn_weights_list[index + depth][list_indices_hidden[layer], :]

        count += 1
        depth = count * 2

    nn_weights_list = np.asarray(nn_weights_list)

    return nn_weights_list


def apply_mask_and_add_noise(nn_weights_list, list_mask, seed):

    count = 0
    depth = count * 2
    for layer in range(len(list_mask)):
        for index in range(3):
            if index == 0:
                # noise the columns
                nn_weights_list[index + depth][:, list_mask[layer]] = \
                    add_noise(nn_weights_list[index + depth][:, list_mask[layer]], 0.5, seed)
            elif index == 1:
                # order columns for bias
                nn_weights_list[index + depth][list_mask[layer]] = \
                    add_noise(nn_weights_list[index + depth][list_mask[layer]], 0.5, seed)
            elif index == 2:
                # order rows
                nn_weights_list[index + depth][list_mask[layer], :] = \
                    add_noise(nn_weights_list[index + depth][list_mask[layer], :], 0.5, seed)

        count += 1
        depth = count * 2

    nn_weights_list = np.asarray(nn_weights_list)

    return nn_weights_list


def crossover_method(weights_one, weights_two, list_corr_matrices, crossover):
    list_ordered_indices_one = []
    list_ordered_indices_two = []
    for index in range(len(list_corr_matrices)):
        corr_matrix_nn = list_corr_matrices[index]

        indices_one, indices_two = bipartite_matching(corr_matrix_nn, crossover)
        list_ordered_indices_one.append(indices_one)
        list_ordered_indices_two.append(indices_two)

    similarity = get_network_similarity(list_corr_matrices, list_ordered_indices_one, list_ordered_indices_two)

    # order the weight matrices

    if crossover == "naive":
        list_ordered_w_one = list(weights_one)
        list_ordered_w_two = list(weights_two)

    else:
        weights_nn_one_copy = list(weights_one)
        weights_nn_two_copy = list(weights_two)
        list_ordered_w_one = apply_mask_to_weights(weights_nn_one_copy, list_ordered_indices_one)
        list_ordered_w_two = apply_mask_to_weights(weights_nn_two_copy, list_ordered_indices_two)

    return list_ordered_w_one, list_ordered_w_two, similarity


def arithmetic_crossover(network_one, network_two, t=0.5):
    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))

    list_weights = []
    for index in range(len(network_one)):
        averaged_weights = (t * network_one[index] + (1 - t) * network_two[index]) * scale_factor
        list_weights.append(averaged_weights)

    return list_weights


def add_noise_to_fittest(network_one, network_two, information_nn_one, information_nn_two, crossover, seed, index):

    t = 0.5
    if crossover == "noise_0.1":
        t = 0.9

    # choose best parent
    best_parent = network_one
    if index == 0:
        pass
    else:
        if np.max(information_nn_one.history["val_loss"]) < np.max(information_nn_two.history["val_loss"]):
            best_parent = network_two

    list_weights = []
    for index in range(len(best_parent)):
        parent_weights = best_parent[index]
        parent_weights = add_noise(parent_weights, t, seed)

        list_weights.append(parent_weights)

    return list_weights


def low_corr_neurons(network_one, network_two, corr_matrices_list, information_nn_one, information_nn_two,
                     seed, index, crossover, threshold):

    # choose best parent
    best_parent = network_one
    best_parent_switch = False
    if index == 0:
        pass
    else:
        if np.max(information_nn_one.history["val_loss"]) < np.max(information_nn_two.history["val_loss"]):
            best_parent = network_two
            best_parent_switch = True

    mask_list = []
    for corr_matrix in corr_matrices_list:
        max_corr_list = []
        for i in range(corr_matrix.shape[0]):
            list_corr = []
            for j in range(corr_matrix.shape[1]):
                if best_parent_switch:
                    corr = np.abs(corr_matrix[i, j])
                else:
                    corr = np.abs(corr_matrix[j, i])

                list_corr.append(corr)
            max_corr_list.append(np.max(list_corr))

        quantile = np.quantile(max_corr_list, threshold)
        mask_array = np.asarray(max_corr_list) >= quantile

        mask_list.append(mask_array)

    if crossover == "pruning_low_corr":
        best_parent = apply_mask_to_weights(best_parent, mask_list)

    elif crossover == "noise_low_corr":
        mask_list = [[not element for element in sublist] for sublist in mask_list]
        best_parent = apply_mask_and_add_noise(best_parent, mask_list, seed)

    return best_parent
