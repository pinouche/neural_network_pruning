import pickle

from utils import add_noise
from compute_mask import apply_mask_to_neurons


def mask_one_action(trained_masked_weight, mask, pruning_method, retraining_method, seed):

    weights_to_retrain = []

    if retraining_method == "fine_tune":
        weights_to_retrain = trained_masked_weight

    elif retraining_method == "add_noise_and_fine_tune":
        if "weights" in pruning_method:
            for index in range(len(mask)):
                weights = trained_masked_weight[index]
                mask_weight = mask[index].astype(bool)

                weights[mask_weight] = add_noise(weights[mask_weight], seed, 0.5)
                weights_to_retrain.append(weights)

        elif "neurons" in pruning_method:
            for index in range(len(trained_masked_weight)):
                weights = trained_masked_weight[index]
                weights = add_noise(weights, seed, 0.5)
                weights_to_retrain.append(weights)

    elif retraining_method == "lottery":
        weights_init = pickle.load(open("weights_init_" + str(seed) + ".pickle", 'rb'))

        if "weights" in pruning_method:
            for index in range(len(mask)):
                weights = weights_init[index]
                mask_weight = mask[index].astype(bool)

                weights = weights*mask_weight
                weights_to_retrain.append(weights)

        elif "neurons" in pruning_method:
            weights_to_retrain = apply_mask_to_neurons(weights_init, mask)

    return weights_to_retrain



