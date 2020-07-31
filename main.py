import numpy as np
import pickle
import os
from timeit import default_timer as timer
import threading
import queue
import multiprocessing
import warnings

import tensorflow as tf
from tensorflow.python import nn
from tensorflow.python import keras

from utils import crossover_method
from utils import arithmetic_crossover
from utils import load_cifar

warnings.filterwarnings("ignore")
np.random.seed(0)


def model_keras(seed, data):
    if data == "mnist":
        input_size = 784
        hidden_size = 512
        output_size = 10

        init_input_weights = tf.random_normal_initializer(mean=0.0, stddev=1 / np.sqrt((input_size + hidden_size) / 2),
                                                          seed=seed)
        init_output_weights = tf.random_normal_initializer(mean=0.0,
                                                           stddev=1 / np.sqrt((hidden_size + output_size) / 2),
                                                           seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=init_input_weights, input_shape=(input_size,)),
            keras.layers.AlphaDropout(0.2, seed=seed),
            # output layer
            keras.layers.Dense(output_size, activation=nn.softmax, use_bias=False,
                               kernel_initializer=init_output_weights)
        ])

    elif data == "cifar10":
        input_size = 3072
        hidden_size = 100
        output_size = 10

        init_input_weights = tf.random_normal_initializer(mean=0.0, stddev=1 / np.sqrt((input_size + hidden_size) / 2),
                                                          seed=seed)
        init_hidden_weights = tf.random_normal_initializer(mean=0.0, stddev=1 / np.sqrt(hidden_size),
                                                           seed=seed)
        init_output_weights = tf.random_normal_initializer(mean=0.0,
                                                           stddev=1 / np.sqrt((hidden_size + output_size) / 2),
                                                           seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=init_input_weights, input_shape=(input_size,)),
            # tf.keras.layers.AlphaDropout(0.1, seed=seed),
            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=init_hidden_weights, input_shape=(input_size,)),
            # tf.keras.layers.AlphaDropout(0.1, seed=seed),
            keras.layers.Dense(hidden_size, activation=nn.selu, use_bias=True,
                               kernel_initializer=init_hidden_weights),
            # output layer
            keras.layers.Dense(output_size, activation=nn.softmax, use_bias=False, kernel_initializer=init_output_weights)
        ])

    else:
        raise Exception("wrong dataset")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, num_threads, work_id, data_struc,
                        parallel="process"):
    # shuffle input data here
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    num_pairs = len(pair_list)
    pairs_to_compute = pair_list[work_id * int(num_pairs / num_threads):(work_id + 1) * int(num_pairs / num_threads)]
    pair = pairs_to_compute[0]

    print("FOR PAIR NUMBER " + str(pair + 1))

    crossover_types = ["safe", "unsafe", "orthogonal", "normed", "naive"]
    result_list = [[] for _ in range(len(crossover_types))]
    count = 0

    for crossover in crossover_types:
        for index in [0]:
            print("EPOCH " + str(index))

            model = model_keras(0, data)
            model_one = model_keras(pair, data)
            model_two = model_keras(pair + num_pairs, data)

            print("one")
            model_one.fit(x_train, y_train, epochs=index, batch_size=128, verbose=False,
                          validation_data=(x_test, y_test))
            weights_nn_one = model_one.get_weights()
            print("three")

            model_two.fit(x_train, y_train, epochs=index, batch_size=128, verbose=False,
                          validation_data=(x_test, y_test))
            weights_nn_two = model_two.get_weights()
            print("five")

            print("crossover method: " + crossover)
            list_ordered_weights_one, list_ordered_weights_two = crossover_method(x_test, weights_nn_one,
                                                                                  weights_nn_two,
                                                                                  crossover)

            print("seven")
            weights_crossover = arithmetic_crossover(list_ordered_weights_one, list_ordered_weights_two, index)
            model.set_weights(weights_crossover)
            # model.evaluate(x_test, y_test)
            model_information_offspring = model.fit(x_train, y_train, epochs=15, batch_size=128,
                                                    verbose=False, validation_data=(x_test, y_test))

            print("eight")
            model_one.set_weights(weights_nn_one)
            model_information_parent_one = model_one.fit(x_train, y_train, epochs=15, batch_size=128,
                                                         verbose=False, validation_data=(x_test, y_test))

            print("nine")
            model_two.set_weights(weights_nn_two)
            model_information_parent_two = model_two.fit(x_train, y_train, epochs=15, batch_size=128,
                                                         verbose=False, validation_data=(x_test, y_test))

            accuracy_best_parent = model_information_parent_one.history["val_acc"]
            if np.max(accuracy_best_parent) < np.max(model_information_parent_two.history["val_acc"]):
                accuracy_best_parent = model_information_parent_two.history["val_acc"]

            result_list[count].append(accuracy_best_parent)
            result_list[count].append(model_information_offspring.history["val_acc"])

        if parallel == "process":
            data_struc[work_id] = result_list
        elif parallel == "thread":
            data_struc.put(result_list)

        count += 1

        print("ten")

        keras.backend.clear_session()


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
                                                                pair_list, num_threads, i, q, parallel_method)) for i in
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
                                                                       pair_list, num_processes, i, return_dict,
                                                                       parallel_method)) for i in range(num_processes)]

        for proc in p:
            proc.start()
        for proc in p:
            proc.join()

        results = return_dict.values()

        end = timer()
        print(end - start)
