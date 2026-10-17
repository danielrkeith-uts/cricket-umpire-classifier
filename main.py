import primary_dataset

import numpy as np

from keypoint_classification import KeypointClassifier


def main():
    primary_dataset_train_annotations = primary_dataset.load_train_annotations()

    keypoint_classifier = KeypointClassifier(
        primary_dataset_train_annotations["X"].shape[1],
        len(primary_dataset_train_annotations["Classes"])
    )

    keypoint_classifier.fit(
        primary_dataset_train_annotations,
        primary_dataset.load_validate_annotations()
    )


if __name__ == "__main__":
    main()
