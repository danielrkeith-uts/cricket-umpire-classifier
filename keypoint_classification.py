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

    def fit(self, train, validate):
        self.model.fit(
            train["X"],
            train["Y"],
            epochs=200,
            batch_size=64,
            callbacks=[_early_stopping],
            validation_data=(validate["X"], validate["Y"])
        )

    def evaluate(self, test):
        return self.model.evaluate(test["X"], test["Y"])
