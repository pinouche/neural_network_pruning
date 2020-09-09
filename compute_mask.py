import numpy as np


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


# prune weights
def apply_mask_to_weights(weights_list, gradients_list, pruning_strategy, threshold):
    pruned_trained_network = []
    mask_list = []

    if pruning_strategy == "pruning_random_weights":  # global
        size = np.sum([np.prod(weight.shape) for weight in weights_list])
        n_zeros = int(size * threshold)
        mask = np.array([0] * n_zeros + [1] * (size - n_zeros))
        np.random.shuffle(mask)

        count = 0
        for weight in weights_list:
            number_weights = np.prod(weight.shape)
            mask_weight = mask[count:number_weights + count]
            mask_weight = mask_weight.reshape(weight.shape)
            weight = weight*mask_weight
            pruned_trained_network.append(weight)
            mask_list.append(mask_weight)
            count += number_weights

        return pruned_trained_network, mask_list

    if pruning_strategy == "pruning_magnitude_weights_local":
        for weight in weights_list:
            score = np.abs(weight)
            quantile = np.quantile(score, threshold)
            mask = np.ones(weight.shape).astype(int)
            mask[score < quantile] = 0
            weight = weight*mask
            pruned_trained_network.append(weight)
            mask_list.append(mask)

        return pruned_trained_network, mask_list

    if pruning_strategy == "pruning_magnitude_weights_global":
        score = [np.abs(weight) for layer in weights_list for weight in layer.flatten()]
        quantile = np.quantile(score, threshold)

        for weight in weights_list:
            score = np.abs(weight)
            mask = np.ones(weight.shape).astype(int)
            mask[score < quantile] = 0
            weight = weight * mask
            pruned_trained_network.append(weight)
            mask_list.append(mask)

        return pruned_trained_network, mask_list

    if pruning_strategy == "pruning_gradient_weights_local":
        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            quantile = np.quantile(score, threshold)
            mask = np.ones(weight.shape).astype(int)
            mask[score < quantile] = 0
            weight = weight*mask
            pruned_trained_network.append(weight)
            mask_list.append(mask)

        return pruned_trained_network, mask_list

    if pruning_strategy == "pruning_gradient_weights_global":
        weights_flatten = np.array([np.abs(weight) for weight_matrix in weights_list for weight in weight_matrix.flatten()])
        gradients_flatten = np.array([gradient for weight_matrix in gradients_list for gradient in weight_matrix.flatten()])
        score = weights_flatten * gradients_flatten
        quantile = np.quantile(score, threshold)

        for index in range(len(weights_list)):
            weight = weights_list[index]
            score = np.abs(weight) * gradients_list[index]
            mask = np.ones(weight.shape).astype(int)
            mask[score < quantile] = 0
            weight = weight * mask
            pruned_trained_network.append(weight)
            mask_list.append(mask)

        return pruned_trained_network, mask_list


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


def prune(weights_trained_original, list_hidden_representation, list_gradient_hidden_layers,
          list_gradient_weights, corr_matrices_list, pruning_strategy, threshold):

    if "pruning" in pruning_strategy:
        # weight-level
        if pruning_strategy in ["pruning_random_weights", "pruning_magnitude_weights_local", "pruning_magnitude_weights_global",
                                "pruning_gradient_weights_local", "pruning_gradient_weights_global"]:

            pruned_network, mask_list = apply_mask_to_weights(weights_trained_original,
                                                              list_gradient_weights, pruning_strategy, threshold)

        # neuron-level (global)
        elif pruning_strategy == "pruning_random_neurons":
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

        # neuron-level
        elif pruning_strategy == "pruning_magnitude_neurons_local":
            mask_list = []
            for hidden_layer in list_hidden_representation:
                score = np.mean(np.abs(hidden_layer), axis=0)
                quantile = np.quantile(score, threshold)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

        elif pruning_strategy == "pruning_magnitude_neurons_global":
            mask_list = []
            score = [neuron for layer in list_hidden_representation for neuron in np.mean(np.abs(layer), axis=0)]
            quantile = np.quantile(score, threshold)

            for hidden_layer in list_hidden_representation:
                score = np.mean(np.abs(hidden_layer), axis=0)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

        elif pruning_strategy == "pruning_gradient_neurons_local":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            for hidden_layer_gradient in list_gradient_hidden_layers:
                score = np.mean(np.abs(hidden_layer_gradient), axis=0)
                quantile = np.quantile(score, threshold)
                mask_array = score >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

        elif pruning_strategy == "pruning_gradient_neurons_global":
            mask_list = []
            list_gradient_hidden_layers = np.mean(np.abs(np.array(list_gradient_hidden_layers)), axis=0)
            score = [neuron for layer in list_gradient_hidden_layers for neuron in layer[0]]
            quantile = np.quantile(score, threshold)

            for hidden_layer_gradient in list_gradient_hidden_layers:
                mask_array = hidden_layer_gradient[0] >= quantile
                mask_list.append(mask_array)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

        # neuron-level (similarity based)
        elif pruning_strategy == "pruning_low_corr_neurons":
            mask_list = compute_mask_low_corr(corr_matrices_list, threshold)
            pruned_network = apply_mask_to_neurons(weights_trained_original, mask_list)

    return pruned_network, mask_list
