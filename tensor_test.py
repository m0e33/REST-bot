"""Executable for testing the functionality of various methods and modules"""

import tensorflow as tf

if __name__ == "__main__":

    w = tf.constant([[4, 3, 2, 1], [8, 7, 6, 5]])

    words = tf.constant([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])

    intermediate = tf.map_fn(
        lambda attention_weight: attention_weight * words,
        w
    )

    bias = tf.constant([[5, 5, 5, 5], [3, 3, 3, 3]])

    output = intermediate + bias
    print(output)
