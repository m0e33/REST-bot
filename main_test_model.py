"""Executable for testing the functionality of various methods and modules"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from data.data_store import DataStore

if __name__ == "__main__":
    TEST_SYMBOL = "AAPL"
    data_store = DataStore([TEST_SYMBOL], "2021-01-01", "2021-04-01")
    prices = data_store.get_price_data(TEST_SYMBOL)
    press_releases = data_store.get_press_release_data(TEST_SYMBOL)

    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=50)
    press_release_texts = [
        pr["text"] for pr in press_releases
    ]  # -> enrich with TYPE (should be known by vocabulary)
    # fill also one NO-EVENT type event
    text_ds = tf.data.Dataset.from_tensor_slices(press_release_texts).batch(128)
    vectorizer.adapt(text_ds)

    output = vectorizer([press_release_texts[0]])
    print(press_release_texts[0])
    print(output.numpy()[0])

    # Filter
    # Fixed Length
