import keypoint_detection
import primary_dataset
import signal_analysis_over_time

import keras
import numpy as np

from keypoint_classification import KeypointClassifier
from signal_analysis_over_time import SignalSequenceClassifier

_SEED = 0
_VALIDATE_FRACTION = 0.15
_WINDOW_LENGTH = 5
_NUM_SEEDS = 3
_FIXED_EPOCHS = 100


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


def test_keypoint_classification_on_unseen_umpires():
    """
    Leave-one-umpire-out: for each umpire, trains on every other umpire and tests on that one.

    Runs three times so the effect of using a pose estimator can be compared:
    1. Train and test on annotated keypoints.
    2. Train on annotated keypoints, test on BlazePose keypoints.
    3. Train and test on BlazePose keypoints.
    """
    dataset = primary_dataset.load_all_annotations()

    print("\nRunning BlazePose on every image...")
    predicted_x, detected = keypoint_detection.detect_body_keypoints(dataset["Image"])
    print(f"BlazePose found a person in {detected.sum()} / {len(detected)} images")

    predicted = dict(dataset, X=predicted_x)

    experiments = [
        ("Annotated keypoints", dataset, dataset, None),
        ("Train annotated, test BlazePose", dataset, predicted, detected),
        ("BlazePose keypoints", predicted, predicted, detected),
    ]

    for name, train_source, test_source, test_detected in experiments:
        print(f"\n\n--- {name.upper()} ---\n")
        _evaluate_on_unseen_umpires(train_source, test_source, test_detected)


def _evaluate_on_unseen_umpires(train_source, test_source, detected=None):
    """
    train_source and test_source must contain the same images in the same order, but may have different X.
    When detected is given, images where no person was detected are left out of training and count as wrong when testing.
    """
    if detected is None:
        detected = np.ones(len(test_source["Y"]), dtype=bool)

    num_classes = len(train_source["Classes"])
    all_true = []
    all_predicted = []

    for umpire in np.unique(train_source["Umpire"]):
        keras.utils.set_random_seed(_SEED)

        is_test = train_source["Umpire"] == umpire
        train, validate = _split_validate(_subset(train_source, ~is_test & detected))
        test = _subset(test_source, is_test)

        keypoint_classifier = KeypointClassifier(train["X"].shape[1], num_classes)
        keypoint_classifier.fit(train, validate, verbose=0)

        # A missed detection is a failure of the whole pipeline, so it is never a correct prediction.
        predicted = np.where(detected[is_test], keypoint_classifier.predict(test["X"]), -1)

        accuracy = np.mean(predicted == test["Y"])
        print(f"{umpire:>10}: {len(test['Y']):4} images, accuracy {accuracy:.3f}, "
              f"macro F1 {_macro_f1(test['Y'], predicted, num_classes):.3f}")

        all_true.append(test["Y"])
        all_predicted.append(predicted)

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)

    print(f"{'All':>10}: {len(all_true):4} images, accuracy {np.mean(all_predicted == all_true):.3f}, "
          f"macro F1 {_macro_f1(all_true, all_predicted, num_classes):.3f}")


def test_signal_analysis_over_time_on_unseen_umpires():
    """
    Compares the LSTM with the single-frame keypoint classifier on the same windows of consecutive video frames,
    using leave-one-umpire-out, repeated with several random seeds.

    The single-frame classifier is scored on the last frame of each window (what it would see in real time), and on
    its predictions averaged over the window. If the LSTM also beats the average, its gain comes from learning how the
    signal moves rather than just from seeing more frames.
    """
    dataset = primary_dataset.load_all_annotations()
    windows = signal_analysis_over_time.build_windows(dataset, _WINDOW_LENGTH)
    print(f"\n{len(signal_analysis_over_time.build_clips(dataset))} clips, "
          f"{len(windows['Y'])} windows of {_WINDOW_LENGTH} frames")

    print("\nRunning BlazePose on every image...")
    predicted_x, _ = keypoint_detection.detect_body_keypoints(dataset["Image"])

    for name, x in [("Annotated keypoints", dataset["X"]), ("BlazePose keypoints", predicted_x)]:
        print(f"\n\n--- {name.upper()}: LSTM VS SINGLE FRAME ---\n")
        _compare_over_time_on_unseen_umpires(dict(dataset, X=x), windows)


def _compare_over_time_on_unseen_umpires(frames, windows):
    num_keypoints = frames["X"].shape[1]
    num_classes = len(frames["Classes"])
    four = primary_dataset.CLASSES.index("Four")
    no_ball = primary_dataset.CLASSES.index("No-ball")

    methods = ["Single frame", "Single frame, averaged", "LSTM"]
    scores = {method: [] for method in methods}

    for seed in range(_NUM_SEEDS):
        all_true = []
        all_predicted = {method: [] for method in methods}

        for umpire in np.unique(windows["Umpire"]):
            keras.utils.set_random_seed(seed)

            is_test = windows["Umpire"] == umpire
            train = _subset(windows, ~is_test)
            test = _subset(windows, is_test)

            # With so few clips, holding some out for early stopping can remove a whole class from training,
            # so both models train for the same fixed number of epochs.
            frame_classifier = KeypointClassifier(num_keypoints, num_classes)
            frame_classifier.fit_fixed_epochs(_frames_in_windows(frames, train), _FIXED_EPOCHS, verbose=0)

            sequence_classifier = SignalSequenceClassifier(_WINDOW_LENGTH, num_keypoints, num_classes)
            sequence_classifier.fit(_window_keypoints(frames, train), _FIXED_EPOCHS, verbose=0)

            test_x = frames["X"][test["Indices"]]
            probabilities = frame_classifier.predict_probabilities(test_x.reshape(-1, num_keypoints, 2))
            probabilities = probabilities.reshape(len(test_x), _WINDOW_LENGTH, num_classes)

            all_predicted["Single frame"].append(np.argmax(probabilities[:, -1], axis=1))
            all_predicted["Single frame, averaged"].append(np.argmax(probabilities.mean(axis=1), axis=1))
            all_predicted["LSTM"].append(sequence_classifier.predict(test_x))
            all_true.append(test["Y"])

        all_true = np.concatenate(all_true)

        for method in methods:
            predicted = np.concatenate(all_predicted[method])
            is_four = all_true == four

            scores[method].append((
                _macro_f1(all_true, predicted, num_classes),
                np.mean(predicted == all_true),
                np.mean(predicted[is_four] == no_ball),
            ))

    print(f"{'Method':>24}  {'Macro F1':>13}  {'Accuracy':>13}  {'Four as No-ball':>15}")
    for method in methods:
        means = np.mean(scores[method], axis=0)
        sds = np.std(scores[method], axis=0)
        print(f"{method:>24}  {means[0]:.3f} ± {sds[0]:.3f}  {means[1]:.3f} ± {sds[1]:.3f}  {means[2]:7.3f} ± {sds[2]:.3f}")


def _frames_in_windows(frames, windows):
    indices = np.unique(windows["Indices"])

    return {"X": frames["X"][indices], "Y": frames["Y"][indices]}


def _window_keypoints(frames, windows):
    return {"X": frames["X"][windows["Indices"]], "Y": windows["Y"]}


def _subset(dataset, mask):
    return {
        key: value if key == "Classes" else value[mask]
        for key, value in dataset.items()
    }


def _split_validate(dataset):
    """
    Holds out a random fraction of frames for early stopping. Both copies of a frame (with and without noise) are kept
    on the same side, so the validation images are not near-duplicates of training images.
    """
    frames = np.unique(dataset["Frame"])
    rng = np.random.default_rng(_SEED)
    validate_frames = rng.choice(frames, size=round(len(frames) * _VALIDATE_FRACTION), replace=False)
    is_validate = np.isin(dataset["Frame"], validate_frames)

    return _subset(dataset, ~is_validate), _subset(dataset, is_validate)


def _macro_f1(y_true, y_predicted, num_classes):
    """
    Mean F1 score over the classes that appear in either y_true or y_predicted (not every umpire has every class).
    Predictions of -1 (no detection) count as wrong but do not add a class.
    """
    f1_scores = []

    for c in range(num_classes):
        true_positives = np.sum((y_predicted == c) & (y_true == c))
        predicted_positives = np.sum(y_predicted == c)
        actual_positives = np.sum(y_true == c)

        if predicted_positives == 0 and actual_positives == 0:
            continue

        f1_scores.append(2 * true_positives / (predicted_positives + actual_positives))

    return np.mean(f1_scores)


def main():
    test_keypoint_classification_from_dataset()
    # test_keypoint_classification_on_unseen_umpires()
    # test_signal_analysis_over_time_on_unseen_umpires()


if __name__ == "__main__":
    main()
