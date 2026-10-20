import primary_dataset, util

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


def test_keypoint_classification_from_dataset():
    train = primary_dataset.load_train_annotations()
    validate = primary_dataset.load_validate_annotations()
    test = primary_dataset.load_test_annotations()
    full_dataset = {
        "X": np.concatenate((train["X"], validate["X"], test["X"])),
        "Y": np.concatenate((train["Y"], validate["Y"], test["Y"])),
        "Classes": train["Classes"],
    }

    keypoint_classifier = KeypointClassifier(
        train["X"].shape[1],
        len(train["Classes"])
    )

    print("\n\n--- TRAINING ---\n\n")

    keypoint_classifier.fit(
        train,
        validate
    )

    print("\n\n--- EVALUATING ---\n\n")

    test_predictions = keypoint_classifier.predict(test["X"])
    full_dataset_predictions = keypoint_classifier.predict(full_dataset["X"])

    print("Evaluation on test data:")
    test_accuracy = np.mean(test["Y"] == test_predictions)
    test_macro_f1 = util.macro_f1(test["Y"], test_predictions, num_classes=len(test["Classes"]))

    print(f"Accuracy: {test_accuracy}")
    print(f"Macro F1: {test_macro_f1}")

    print("\nEvaluation on entire dataset:")
    full_dataset_accuracy = np.mean(full_dataset["Y"] == full_dataset_predictions)
    full_dataset_macro_f1 = util.macro_f1(full_dataset["Y"], full_dataset_predictions, num_classes=len(test["Classes"]))

    print(f"Accuracy: {full_dataset_accuracy}")
    print(f"Macro F1: {full_dataset_macro_f1}")
