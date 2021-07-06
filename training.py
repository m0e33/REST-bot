"""Executable for testing the functionality of various methods and modules"""

import tensorflow as tf
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from model.configuration import TrainConfiguration, HyperParameterConfiguration
from model.model import RESTNet
from tqdm import tqdm


if __name__ == "__main__":
    data_cfg = DataConfiguration(
        symbols=["AAPL", "ACN", "CDW", "NFLX"],
        start="2020-12-29",
        end="2021-04-06",
        feedback_metrics=["open", "close", "high", "low", "vwap"],
        stock_context_days=6,
    )

    train_cfg = TrainConfiguration()
    hp_cfg = HyperParameterConfiguration()

    data_store = DataStore(data_cfg)
    data_store.build()

    prepro = Preprocessor(data_store, data_cfg, train_cfg)
    prepro.build_events_data_with_gt()

    train_ds = prepro.get_train_ds()
    val_ds = prepro.get_val_ds()
    test_ds = prepro.get_test_ds()

    model = RESTNet(hp_cfg)
    model.run_eagerly = True

    num_epochs = 201

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss(model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    train_loss_results = []
    train_accuracy_results = []
    print("Started training")
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in tqdm(train_ds):
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))
        print("epoch done")
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
