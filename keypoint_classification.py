import numpy as np
from keras.src.callbacks import EarlyStopping
from tensorflow.keras import models, layers

_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


class KeypointClassifier:
    def __init__(self, num_keypoints, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=(num_keypoints, 2)),
            layers.Flatten(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),

            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.accuracy = None

    def fit(self, train, validate, verbose="auto"):
        self.model.fit(
            train["X"],
            train["Y"],
            epochs=200,
            batch_size=64,
            callbacks=[_early_stopping],
            validation_data=(validate["X"], validate["Y"]),
            verbose=verbose
        )

    def fit_fixed_epochs(self, train, epochs, verbose="auto"):
        """
        Trains without early stopping, for when there is too little data to hold out a validation set.
        """
        self.model.fit(
            train["X"],
            train["Y"],
            epochs=epochs,
            batch_size=64,
            verbose=verbose
        )

    def evaluate(self, test):
        return self.model.evaluate(test["X"], test["Y"])

    def predict(self, x):
        return np.argmax(self.predict_probabilities(x), axis=1)

    def predict_probabilities(self, x):
        return self.model.predict(x, verbose=0)
