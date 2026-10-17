import primary_dataset

from keypoint_classification import KeypointClassifier


def main():
    keypoint_classifier = KeypointClassifier()

    keypoint_classifier.fit(
        primary_dataset.load_train_annotations(),
        primary_dataset.load_validate_annotations()
    )


if __name__ == "__main__":
    main()
