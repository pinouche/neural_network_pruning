import pickle
from utils import add_noise


def mask_one_action(trained_masked_weight, initial_weights, mask, seed, retraining_strategy = "fine_tune"):

    if mask is None:
        mask = [1] * len(trained_masked_weight)

    weights_to_retrain = []

    if retraining_strategy == "fine_tune":
        weights_to_retrain = trained_masked_weight

    elif retraining_strategy == "add_noise_and_fine_tune":
        for index in range(len(trained_masked_weight)):
            weights = trained_masked_weight[index]
            mask_weight = mask[index]

            weights[mask_weight] = add_noise(weights[mask_weight], 0.5, seed)
            weights_to_retrain.append(weights)

    elif retraining_strategy == "lottery":
        weights_init = pickle.load(open("weights_init_" + str(seed) + ".pickle", 'rb'))


