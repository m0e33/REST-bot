"""Layer"""

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU


class TypeSpecificEncoder(keras.layers.Layer):
    """Layer docstring"""

    def __init__(self, num_heads):
        super().__init__()
        self._num_heads = num_heads
        self.leaky_relu = LeakyReLU()
        self._event_embedding_shape = None
        self._word_embedding_size = None
        self.w = None
        self.b = None

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""
        self._word_embedding_size = input_shape[4]
        self._event_embedding_shape = self._word_embedding_size * self._num_heads
        # pylint: disable=invalid-name
        self.w = self.add_weight(
            shape=(self._num_heads, self._word_embedding_size),
            initializer="random_normal",
            trainable=True,
        )

        # pylint: disable=invalid-name
        self.b = self.add_weight(
            shape=(self._num_heads, self._word_embedding_size),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        """The layers forward pass"""
        return self._recursive_map(inputs)

    def _recursive_map(self, inputs):
        if tf.rank(inputs).numpy() <= 2:
            return self._attention_map(inputs)
        output_shape = self._get_recursive_output_shape(inputs)
        return tf.map_fn(
            self._recursive_map,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=output_shape),
        )

    def _attention_map(self, event):
        if tf.math.count_nonzero(event) == 0:
            return tf.zeros(shape=self._event_embedding_shape)
        type_embedding = event[0]
        words_embedding = event[1:]

        # the following lambda executes the computations that have to be done per head.
        # In order to iterate over weights and biases of every head simultaneously, we have to carry
        # the bias values through the lambda and the return value
        weighted_sum_vectors, _ = tf.map_fn(
            lambda attention_parameters: (
                self.calculate_normalized_attention(
                    attention_parameters, words_embedding, type_embedding
                ),
                attention_parameters[1],
            ),
            (self.w, self.b),
        )

        # concatenate all weighted sum vectors of each head for a final event embedding
        event_embedding = tf.reshape(
            weighted_sum_vectors, [self._num_heads * self._word_embedding_size]
        )

        return event_embedding

    def calculate_normalized_attention(
        self, attention_parameters, word_embeddings, event_type_embedding
    ):
        """Attention implementation for one head"""

        attention_weights = attention_parameters[0]
        attention_bias = attention_parameters[1]

        # ====
        # paper equation 1 of attention mechanism
        # ====
        leaky_relu_in = attention_weights * word_embeddings + attention_bias
        hidden_representation = self.leaky_relu(leaky_relu_in)

        # ===
        # paper equation 2 of attention mechanism
        # ===
        # In order for the simultaneous matmul with all words of the event to work
        # the shape of the type embedding has to be adjusted
        event_type_embedding = tf.broadcast_to(
            event_type_embedding, hidden_representation.shape
        )

        # Adjusted shape matmul yields shape (49, 49) containing 49 rows of the same values.
        # We need only one row of this
        matmul = tf.matmul(event_type_embedding, tf.transpose(hidden_representation))[0]

        attention_values = tf.math.exp(matmul)
        attention_values_sum = tf.math.reduce_sum(attention_values)
        normalized_attention_values = attention_values / attention_values_sum

        # ===
        # paper equation 3 of attention mechanism
        # ===
        # this twists my brain on a regular basis:
        # every word has now one normalized attention value, which has to be multiplied
        # with every value in the 300 long word embedding.
        # This has to be done with every word.
        # This happens with the following two lines.
        # The output shape of weighted_word_embedding is still (49 * 300)
        normalized_attention_values = tf.broadcast_to(
            tf.expand_dims(normalized_attention_values, axis=1), word_embeddings.shape
        )
        weighted_word_embedding = word_embeddings * normalized_attention_values

        weighted_sum_vector = tf.reduce_sum(weighted_word_embedding, axis=0)

        return weighted_sum_vector

    def _get_recursive_output_shape(self, inputs):
        """Removes the last two dimensions of the input and replaces them with a single dimension
        representing the event_embedding shape. Also, the first dimension has to be removed
        for whatever reason"""

        return tuple(inputs.shape[1:-2] + self._event_embedding_shape)
