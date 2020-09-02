import numpy as np
from keras.models import load_model

from utils import add_noise


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
    mask_list = []

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
            weight = weight*mask
            pruned_network.append(weight)
            mask_list.append(mask_weight)
            count += number_weights

        return pruned_network, mask_list

    if crossover == "pruning_magnitude_weights_local":
        for weight in weights_list:
            score = np.abs(weight)
            quantile = np.quantile(score, threshold)
            mask = np.ones(weight.shape)
            mask[score < quantile] = 0
            weight = weight*mask
            pruned_network.append(weight)
            mask_list.append(mask)

        return pruned_network, mask_list

    if crossover == "pruning_magnitude_weights_global":
        score = [np.abs(weight) for layer in weights_list for weight in layer.flatten()]
        quantile = np.quantile(score, threshold)

        for weight in weights_list:
            score = np.abs(weight)
            mask = np.ones(weight.shape)
            mask[score < quantile] = 0
            weight = weight * mask
            pruned_network.append(weight)
            mask_list.append(mask)

        return pruned_network, mask_list

    if crossover == "pruning_gradient_weights_local":
        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            quantile = np.quantile(score, threshold)
            mask = np.ones(weight.shape)
            mask[score < quantile] = 0
            weight = weight*mask
            pruned_network.append(weight)
            mask_list.append(mask)

        return pruned_network, mask_list

    if crossover == "pruning_gradient_weights_global":
        weights_flatten = [np.abs(weight) for weight_matrix in weights_list for weight in weight_matrix.flatten()]
        gradients_flatten = [gradient for weight_matrix in gradients_list for gradient in weight_matrix.flatten()]
        score = weights_flatten * gradients_flatten
        quantile = np.quantile(score, threshold)

        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            mask = np.ones(weight.shape)
            mask[score < quantile] = 0
            weight = weight * mask
            pruned_network.append(weight)
            mask_list.append(mask)

        return pruned_network, mask_list


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
            pruned_network, mask_list = apply_mask_to_weights(weights_trained_original, gradients_list, crossover, threshold)

            return pruned_network, mask_list

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

            return pruned_network, None

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

            return pruned_network, None

        if crossover == "pruning_magnitude_neurons_global":
            mask_list = []
            score = [neuron for layer in list_hidden_representation for neuron in np.mean(np.abs(layer), axis=0)]
            quantile = np.quantile(score, threshold)

            for hidden_layer in list_hidden_representation:
                score = np.mean(np.abs(hidden_layer), axis=0)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network, None

        elif crossover == "pruning_gradient_neurons_local":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            for hidden_layer_gradient in list_gradient_hidden_layers:
                score = np.mean(np.abs(hidden_layer_gradient), axis=0)
                quantile = np.quantile(score, threshold)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network, None

        elif crossover == "pruning_gradient_neurons_global":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            score = [neuron for layer in list_gradient_hidden_layers for neuron in layer[0]]
            quantile = np.quantile(score, threshold)

            for hidden_layer_gradient in list_gradient_hidden_layers:
                mask_array = hidden_layer_gradient[0] >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network, None

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_fine_tune":
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

            return pruned_network, None

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_lottery":
            weights_init_original = load_model("network_init_" + str(seed) + ".hd5")
            weights_init_original = weights_init_original.get_weights()
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            weights_init_original = apply_mask_to_neurons(weights_init_original, mask_list)

            return weights_init_original, None

        # neuron-level (similarity based)
        elif crossover == "pruning_low_corr_add_noise":
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            weights_trained_original = apply_mask_to_neurons(weights_trained_original, mask_list)
            pruned_network = []
            for index in range(len(weights_trained_original)):
                weights = weights_trained_original[index]
                weights = add_noise(weights, 0.5, seed)
                pruned_network.append(weights)

            return pruned_network, None
