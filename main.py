from timeit import default_timer as timer
import threading
import queue
import multiprocessing
import warnings
import pickle

import tensorflow as tf
import keras

from utils import keras_get_gradient_weights
from utils import keras_get_hidden_layers
from utils import get_gradients_hidden_layers
from utils import get_corr
from utils import load_cifar

from feed_forward import CustomSaver
from feed_forward import model_keras
from feed_forward import init_keras_variables_session

from compute_mask import prune
from retraining_strategies import mask_one_action

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def crossover_offspring(data, x_train, y_train, x_test, y_test, work_id, data_struc, parallel="process"):
    # shuffle input data here
    # np.random.seed(work_id)
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    print("FOR PAIR NUMBER " + str(work_id + 1))

    pruning_strategies = ["pruning_low_corr_neurons", "pruning_random_weights", "pruning_magnitude_weights_local",
                          "pruning_magnitude_weights_global", "pruning_gradient_weights_local", "pruning_gradient_weights_global",
                          "pruning_random_neurons", "pruning_magnitude_neurons_local", "pruning_magnitude_neurons_global",
                          "pruning_gradient_neurons_local", "pruning_gradient_neurons_global"]

    retraining_strategies = ["fine_tune", "add_noise_and_fine_tune", "lottery"]

    result_list = [[] for _ in range(len(pruning_strategies))]
    batch_size_original = 256
    quantile = 0.5
    total_training_epoch = 20
    epoch_list = [20]

    print("one")
    model_original = model_keras(work_id, data)
    init_keras_variables_session()

    weights_init = model_original.get_weights()
    pickle.dump(weights_init, open("weights_init_" + str(work_id) + ".pickle", 'wb'))

    save_callback = CustomSaver(epoch_list)
    model_information_original_network = model_original.fit(x_train, y_train, epochs=total_training_epoch,
                                                            batch_size=batch_size_original,
                                                            verbose=False,
                                                            validation_data=(x_test, y_test), callbacks=[save_callback])

    count = 0
    for pruning_method in pruning_strategies:
        print("Pruning method: " + pruning_method)
        for retraining_method in retraining_strategies:
            print("Retraining method: " + retraining_method)

            # get the parent weights at the corresponding epoch
            weights_trained_original = pickle.load(open("trained_weights_" + str(epoch_list[0]) + ".pickle", "rb"))
            original_network = model_keras(work_id, data, None, weights_trained_original, None)

            list_gradient_weight = keras_get_gradient_weights(original_network, x_test)
            list_gradient_hidden_layers = get_gradients_hidden_layers(original_network, x_test)
            list_hidden_representation = keras_get_hidden_layers(original_network, x_test)

            list_corr_matrices = get_corr(list_hidden_representation)

            # pruning strategy
            weights_pruned, mask = prune(weights_trained_original, list_hidden_representation, list_gradient_hidden_layers,
                                         list_gradient_weight, list_corr_matrices, pruning_method, quantile)
            # re-train strategy
            weights_pruned = mask_one_action(weights_pruned, mask, pruning_method, retraining_method, work_id)

            # get the new size of the networks, and if pruning at neuron-level, we do not need the mask
            pruned_hidden_layer_size = [weight.shape[0] for weight in weights_pruned if len(weight.shape) == 1]
            if "neurons" in pruning_method:
                mask = None

            model_pruned = model_keras(0, data, pruned_hidden_layer_size, weights_pruned, mask)
            model_information_pruned = model_pruned.fit(x_train, y_train,
                                                        epochs=total_training_epoch,
                                                        batch_size=batch_size_original,
                                                        verbose=False, validation_data=(x_test, y_test))

            result_list[count].append(model_information_pruned.history["val_loss"])

            keras.backend.clear_session()
        result_list[count].append(model_information_original_network.history["val_loss"])
        count += 1

    if parallel == "process":
        data_struc[str(work_id) + "_performance"] = result_list
    elif parallel == "thread":
        data_struc.put(result_list)


if __name__ == "__main__":

    data = "cifar10"
    x_train, x_test, y_train, y_test = load_cifar()
    parallel_method = "process"

    if parallel_method == "thread":
        num_threads = 1

        start = timer()

        q = queue.Queue()

        pair_list = [pair for pair in range(num_threads)]

        t = [threading.Thread(target=crossover_offspring, args=(data, x_train, y_train, x_test, y_test,
                                                                pair_list, i, q, parallel_method)) for i in
             range(num_threads)]

        for thread in t:
            thread.start()

        results = [q.get() for _ in range(num_threads)]

        # Stop these threads
        for thread in t:
            thread.stop = True

        end = timer()
        print(end - start)

    elif parallel_method == "process":
        num_processes = 2

        start = timer()

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        pair_list = [pair for pair in range(num_processes)]

        p = [multiprocessing.Process(target=crossover_offspring, args=(data, x_train, y_train, x_test, y_test,
                                                                       worker_id, return_dict,
                                                                       parallel_method)) for worker_id in
             range(num_processes)]

        for proc in p:
            proc.start()
        for proc in p:
            proc.join()

        results = return_dict.values()
        pickle.dump(results, open("pruning_results.pickle", 'wb'))

        end = timer()
        print(end - start)
