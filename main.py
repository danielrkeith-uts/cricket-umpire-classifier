import primary_dataset

import numpy as np

from keypoint_classification import KeypointClassifier


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

    print("Evaluation on test data:")
    keypoint_classifier.evaluate(test)

    print("\nEvaluation on entire dataset:")
    keypoint_classifier.evaluate(full_dataset)


def main():
    test_keypoint_classification_from_dataset()


if __name__ == "__main__":
    main()
