import numpy as np
from timeit import default_timer as timer
import threading
import queue
import multiprocessing
import warnings

# import tensorflow as tf
from tensorflow.python import nn
# from tensorflow.python import keras
from keras.models import load_model
import keras

from utils import get_intermediate_layers
from utils import get_corr
from utils import crossover_method
from utils import arithmetic_crossover
from utils import load_cifar
from utils import add_noise_to_fittest
from utils import low_corr_neurons

warnings.filterwarnings("ignore")


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, epoch_list, parent_id):
        self.epoch_list = epoch_list
        self.parent_id = parent_id

    def on_epoch_end(self, epoch, logs={}):
        if epoch in self.epoch_list:
            self.model.save("model_" + self.parent_id + "_epoch" + str(epoch) + ".hd5")


def model_keras(seed, data):
    if data == "mnist":
        input_size = 784
        hidden_size = 512
        output_size = 10

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=initializer, input_shape=(input_size,)),
            keras.layers.AlphaDropout(0.2, seed=seed),
            # output layer
            keras.layers.Dense(output_size, activation=nn.softmax, use_bias=False,
                               kernel_initializer=initializer)
        ])

    elif data == "cifar10":
        input_size = 3072
        hidden_size = 100
        output_size = 10

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=initializer, input_shape=(input_size,)),

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=initializer, input_shape=(input_size,)),

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=initializer),
            # output layer
            keras.layers.Dense(output_size, activation=nn.softmax, use_bias=False,
                               kernel_initializer=initializer)
        ])

    else:
        raise Exception("wrong dataset")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc,
                        parallel="process"):
    # shuffle input data here
    # np.random.seed(work_id)
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    num_pairs = len(pair_list)
    pair_id = work_id

    print("FOR PAIR NUMBER " + str(pair_id + 1))

    # crossover_types = ["safe", "unsafe", "orthogonal", "normed", "naive", "noise_0.5", "noise_0.1", "pruning_low_corr,"
    #                                                                                                "noise_low_corr"]

    crossover_types = ["noise_low_corr"]
    result_list = [[] for _ in range(len(crossover_types))]
    similarity_list = [[] for _ in range(len(crossover_types))]
    total_training_epoch = 2
    epoch_list = [1]

    model_one = model_keras(pair_id, data)
    model_two = model_keras(pair_id + num_pairs, data)

    print("one")
    save_callback = CustomSaver(epoch_list, "parent_one")
    model_information_parent_one = model_one.fit(x_train, y_train, epochs=total_training_epoch, batch_size=256,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])
    print("two")

    save_callback = CustomSaver(epoch_list, "parent_two")
    model_information_parent_two = model_two.fit(x_train, y_train, epochs=total_training_epoch, batch_size=256,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])
    print("three")

    accuracy_best_parent = model_information_parent_one.history["val_loss"]
    if np.max(accuracy_best_parent) < np.max(model_information_parent_two.history["val_loss"]):
        accuracy_best_parent = model_information_parent_two.history["val_loss"]

    for crossover_time in epoch_list:
        count = 0
        for crossover in crossover_types:

            # get the parent weights at the corresponding epoch
            parent_one = load_model("model_parent_one_epoch" + str(crossover_time) + ".hd5")
            weights_nn_one = parent_one.get_weights()

            parent_two = load_model("model_parent_two_epoch" + str(crossover_time) + ".hd5")
            weights_nn_two = parent_two.get_weights()

            print("crossover method: " + crossover)
            list_ordered_weights_one, list_ordered_weights_two = weights_nn_one, weights_nn_two
            list_hidden_representation_one = get_intermediate_layers(x_test, list_ordered_weights_one)
            list_hidden_representation_two = get_intermediate_layers(x_test, list_ordered_weights_two)
            list_corr_matrices = get_corr(list_hidden_representation_one, list_hidden_representation_two)

            if crossover in ["safe", "unsafe", "orthogonal", "normed", "naive"]:
                list_ordered_weights_one, list_ordered_weights_two, parents_similarity = crossover_method(
                    list_ordered_weights_one, list_ordered_weights_two, list_corr_matrices, crossover)

            print("seven")
            if crossover in ["noise_0.5", "noise_0.1"]:
                weights_crossover = add_noise_to_fittest(list_ordered_weights_one, list_ordered_weights_two,
                                                         model_information_parent_one, model_information_parent_two,
                                                         crossover,
                                                         pair_id, crossover_time)

            elif crossover in ["pruning_low_corr", "noise_low_corr"]:
                weights_crossover = low_corr_neurons(list_ordered_weights_one, list_ordered_weights_two,
                                                    list_corr_matrices,
                                                    model_information_parent_one, model_information_parent_two,
                                                    pair_id, crossover_time, crossover, 0.5)

            else:
                weights_crossover = arithmetic_crossover(list_ordered_weights_one, list_ordered_weights_two)

            model_offspring = model_keras(0, data)

            model_offspring.set_weights(weights_crossover)
            model_information_offspring = model_offspring.fit(x_train, y_train,
                                                              epochs=total_training_epoch - crossover_time,
                                                              batch_size=256,
                                                              verbose=False, validation_data=(x_test, y_test))

            result_list[count].append(accuracy_best_parent)
            result_list[count].append(model_information_offspring.history["val_loss"])

            if crossover not in ["noise_0.5", "noise_0.1", "pruning_low_corr", "noise_low_corr"]:
                similarity_list[count].append(parents_similarity)

            keras.backend.clear_session()
            count += 1

    if parallel == "process":
        data_struc[str(work_id) + "_performance"] = result_list
        data_struc[str(work_id) + "_similarity"] = similarity_list
    elif parallel == "thread":
        data_struc.put(result_list)

    print("ten")


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
                                                                       pair_list, i, return_dict,
                                                                       parallel_method)) for i in range(num_processes)]

        for proc in p:
            proc.start()
        for proc in p:
            proc.join()

        results = return_dict.values

        end = timer()
        print(end - start)
