"""Executable for testing the functionality of various methods and modules"""

import tensorflow as tf

if __name__ == "__main__":
    inputs = tf.random.uniform(shape=[3, 4, 41, 600])
    _num_days = 2

    print("(Letzter) 3. Tag, 0. Symbol, 0. Event:")
    print(inputs[2][0][0][:5])

    data_last_days = tf.slice(inputs, begin=[inputs.shape[0] - _num_days, 0, 0, 0],
                              size=[_num_days, inputs.shape[1], inputs.shape[2], inputs.shape[3]])

    print("Slice -> shape")
    print(data_last_days.shape)

    print("Transpose -> shape")
    transposed_days = tf.transpose(data_last_days, perm=[1, 0, 2, 3])
    print(transposed_days.shape)

    print("--- check if this is same")
    print("(Letzter) 2. Tag, 0. Symbol, 0. Event:")
    print(transposed_days[0][1][0][:5])

    flattend_days = tf.reshape(transposed_days,
                               shape=[transposed_days.shape[0], transposed_days.shape[1] * transposed_days.shape[2],
                                      transposed_days.shape[3]])

    print("Reshape -> shape")
    print(flattend_days.shape)

    print("--- check if this is same")
    print("(Letzter) 41. Tag, 0. Symbol, 0. Event:")
    print(flattend_days[0][41][:5])