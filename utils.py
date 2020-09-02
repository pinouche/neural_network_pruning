import numpy as np
import pickle
from keras.models import load_model
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


def add_noise(parent_weights, t, seed):
    np.random.seed(seed)

    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))
    mean_parent, std_parent = 0.0, np.std(parent_weights)

    weight_noise = np.random.normal(loc=mean_parent, scale=std_parent, size=parent_weights.shape)
    parent_weights = (t * parent_weights + (1 - t) * weight_noise) * scale_factor

    return parent_weights


# Algorithm 2 (apply mask to neurons)
def apply_mask_to_neurons(nn_weights_list, neurons_mask):
    count = 0
    depth = count * 2
    for layer in range(len(neurons_mask)):
        for index in range(3):
            if index == 0:
                # order columns for weights
                nn_weights_list[index + depth] = nn_weights_list[index + depth][:, neurons_mask[layer]]
            elif index == 1:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][
                    neurons_mask[layer]]  # order columns for bias
            elif index == 2:
                # order rows
                nn_weights_list[index + depth] = nn_weights_list[index + depth][neurons_mask[layer], :]

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


# prune weights
def apply_mask_to_weights(weights_list, gradients_list, crossover, threshold):
    pruned_network = []

    if crossover == "pruning_random_weights": #global
        size = np.sum([np.prod(weight.shape) for weight in weights_list])
        n_zeros = int(size * threshold)
        mask = np.array([0] * n_zeros + [1] * (size - n_zeros))
        np.random.shuffle(mask)

        count = 0
        for weight in weights_list:
            number_weights = np.prod(weight.shape)
            mask_weight = mask[count:number_weights + count]
            mask_weight = mask_weight.reshape(weight.shape)
            mask_weight = mask_weight.astype(bool)
            weight[mask_weight] = 0
            pruned_network.append(weight)
            count += number_weights

        return pruned_network

    if crossover == "pruning_magnitude_weights_local":
        for weight in weights_list:
            score = np.abs(weight)
            quantile = np.quantile(score, threshold)
            weight[score < quantile] = 0
            pruned_network.append(weight)

        return pruned_network

    if crossover == "pruning_magnitude_weights_global":
        score = [np.abs(weight) for layer in weights_list for weight in layer.flatten()]
        quantile = np.quantile(score, threshold)

        for weight in weights_list:
            score = np.abs(weight)
            weight[score < quantile] = 0
            pruned_network.append(weight)

        return pruned_network

    if crossover == "pruning_gradient_weights_local":
        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            quantile = np.quantile(score, threshold)
            weight[score < quantile] = 0
            pruned_network.append(weight)

        return pruned_network

    if crossover == "pruning_gradient_weights_global":
        weights_flatten = [np.abs(weight) for weight_matrix in weights_list for weight in weight_matrix.flatten()]
        gradients_flatten = [gradient for weight_matrix in gradients_list for gradient in weight_matrix.flatten()]
        score = weights_flatten * gradients_flatten
        quantile = np.quantile(score, threshold)

        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            weight[score < quantile] = 0
            pruned_network.append(weight)

        return pruned_network


def compute_mask_low_corr(corr_matrices_list, threshold):
    mask_list = []
    for corr_matrix in corr_matrices_list:
        max_corr_list = []
        for i in range(corr_matrix.shape[0]):
            list_corr = np.abs(corr_matrix[i, :])
            list_corr.sort()
            # take the second highest correlation since the highest = 1 (with itself)
            max_corr_list.append(list_corr[-2])

        quantile = np.quantile(max_corr_list, threshold)
        mask_array = np.asarray(max_corr_list) >= quantile

        mask_list.append(mask_array)

    return mask_list


def prune(weights_trained_original, list_hidden_representation, list_gradient_hidden_layers, gradients_list,
          corr_matrices_list, seed, crossover, threshold):

    if "pruning" in crossover:
        # weight-level
        if crossover in ["pruning_random_weights", "pruning_magnitude_weights_local", "pruning_magnitude_weights_global",
                         "pruning_gradient_weights_local", "pruning_gradient_weights_global"]:
            pruned_network = apply_mask_to_weights(weights_trained_original, gradients_list, crossover, threshold)

            return pruned_network

        # neuron-level (global)
        elif crossover == "pruning_random_neurons":
            mask_list = []
            size_list = [weight.shape[0] for weight in weights_trained_original if len(weight.shape) == 1]
            total_number_neurons = np.sum(size_list)
            n_zeros = int(total_number_neurons * threshold)
            mask = np.array([0] * n_zeros + [1] * (total_number_neurons - n_zeros))
            np.random.shuffle(mask)

            count = 0
            for layer_size in size_list:
                mask_weight = mask[count:layer_size + count]
                mask_weight = mask_weight.astype(bool)
                count += layer_size
                mask_list.append(mask_weight)

            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        # neuron-level
        elif crossover == "pruning_magnitude_neurons_local":
            mask_list = []
            for hidden_layer in list_hidden_representation:
                score = np.mean(np.abs(hidden_layer), axis=0)
                print(score.shape)
                quantile = np.quantile(score, threshold)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        if crossover == "pruning_magnitude_neurons_global":
            mask_list = []
            score = [neuron for layer in list_hidden_representation for neuron in np.mean(np.abs(layer), axis=0)]
            quantile = np.quantile(score, threshold)

            for hidden_layer in list_hidden_representation:
                score = np.mean(np.abs(hidden_layer), axis=0)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        elif crossover == "pruning_gradient_neurons_local":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            for hidden_layer_gradient in list_gradient_hidden_layers:
                score = np.mean(np.abs(hidden_layer_gradient), axis=0)
                quantile = np.quantile(score, threshold)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        elif crossover == "pruning_gradient_neurons_global":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            score = [neuron for layer in list_gradient_hidden_layers for neuron in layer[0]]
            quantile = np.quantile(score, threshold)

            for hidden_layer_gradient in list_gradient_hidden_layers:
                mask_array = hidden_layer_gradient[0] >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_fine_tune":
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_lottery":
            weights_init_original = load_model("network_init_" + str(seed) + ".hd5")
            weights_init_original = weights_init_original.get_weights()
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            weights_init_original = apply_mask_to_neurons(weights_init_original, mask_list)

            return weights_init_original

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_add_noise":
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            weights_trained_original = apply_mask_to_neurons(weights_trained_original, mask_list)
            pruned_network = []
            for index in range(len(weights_trained_original)):
                weights = weights_trained_original[index]
                weights = add_noise(weights, 0.5, seed)
                pruned_network.append(weights)

            return pruned_network
