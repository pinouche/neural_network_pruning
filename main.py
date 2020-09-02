import numpy as np
from timeit import default_timer as timer
import threading
import queue
import multiprocessing
import warnings
import pickle

import keras
from keras.models import load_model

from utils import keras_get_gradient_weights
from utils import keras_get_hidden_layers
from utils import get_gradients_hidden_layers
from utils import get_corr
from utils import load_cifar
from utils import prune

from build_model import CustomSaver
from build_model import model_keras

warnings.filterwarnings("ignore")

#model_keras(seed, data, new_hidden_size_list=None, weights_list=None, mask_list=None)


def crossover_offspring(data, x_train, y_train, x_test, y_test, work_id, data_struc, parallel="process"):
    # shuffle input data here
    # np.random.seed(work_id)
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    print("FOR PAIR NUMBER " + str(work_id + 1))

    # pruning_types = ["pruning_low_corr_neurons_fine_tune", "pruning_low_corr_neurons_add_noise",
    # "pruning_low_corr_neurons_lottery", pruning_random_weights", "pruning_magnitude_weights_local",
    # "pruning_magnitude_weights_global", "pruning_gradient_weights_local", "pruning_gradient_weights_global",
    # "pruning_random_neurons", "pruning_magnitude_neurons_local", "pruning_magnitude_neurons_global".
    # "pruning_gradient_neurons_local", "pruning_gradient_neurons_global"]
    pruning_types = ["pruning_magnitude_neurons_global"]

    result_list = [[] for _ in range(len(pruning_types))]
    batch_size_original = 256
    quantile = 0.5
    total_training_epoch = 20
    epoch_list = [20]

    print("one")
    model_original = model_keras(work_id, data)
    model_original.save("network_init_" + str(work_id) + ".hd5")

    save_callback = CustomSaver(epoch_list)
    model_information_original_network = model_original.fit(x_train, y_train, epochs=total_training_epoch,
                                                            batch_size=batch_size_original,
                                                            verbose=False,
                                                            validation_data=(x_test, y_test), callbacks=[save_callback])
    print("two")

    count = 0
    for pruning_method in pruning_types:
        # get the parent weights at the corresponding epoch
        original_network = load_model("original_network_" + str(epoch_list[0]) + ".hd5")
        weights_trained_original = original_network.get_weights()

        print("crossover method: " + pruning_method)
        list_gradient_weight = keras_get_gradient_weights(original_network, x_test)
        list_gradient_hidden_layers = get_gradients_hidden_layers(original_network, x_test)
        list_hidden_representation = keras_get_hidden_layers(original_network, x_test)
        list_corr_matrices = get_corr(list_hidden_representation)

        weights_pruned = prune(weights_trained_original, list_hidden_representation,
                               list_gradient_hidden_layers,
                               list_gradient_weight,
                               list_corr_matrices, work_id,
                               pruning_method, quantile)

        # get the new size of the networks
        pruned_hidden_layer_size = [weight.shape[0] for weight in weights_pruned if len(weight.shape) == 1]
        print(pruned_hidden_layer_size)

        model_pruned = model_keras(0, data, pruned_hidden_layer_size)

        model_pruned.set_weights(weights_pruned)
        model_information_pruned = model_pruned.fit(x_train, y_train,
                                                    epochs=int(total_training_epoch / 2),
                                                    batch_size=batch_size_original,
                                                    verbose=False, validation_data=(x_test, y_test))

        result_list[count].append(model_information_original_network.history["val_loss"])
        result_list[count].append(model_information_pruned.history["val_loss"])

        keras.backend.clear_session()
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
        num_processes = 1

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

        results = return_dict.values
        pickle.dump(results, open("crossover_results.pickle", 'wb'))

        end = timer()
        print(end - start)
