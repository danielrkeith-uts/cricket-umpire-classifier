"""
Evaluation of the keypoint detection stage on its own, measured against the hand-marked keypoints in the primary
dataset (Project3020, 2025).

Four experiments are run:
1. Estimator comparison: BlazePose against three sizes of YOLO11-pose, for accuracy, detection rate and speed.
2. Oracle cropping: the full frame against a crop around the hand-marked umpire bounding box. This is the best case of
   what the umpire detection stage could provide, not a test of the full pipeline.
3. Image quality: how accuracy changes when images are degraded in ways common in amateur footage (noise, low light,
   blur, low resolution), and whether classic image processing filters recover the lost accuracy.
4. Dataset noise: the clean copy of each frame against the copy with Roboflow's salt-and-pepper noise.
5. Classification: how each estimator's keypoints affect the keypoint classifier on unseen umpires.

Accuracy is reported as:
- PCK@a (percentage of correct keypoints): a predicted keypoint is correct if it lies within a * (umpire height) of
  the hand-marked keypoint. The umpire height is the height of the annotated bounding box. Images where no person is
  found count as every keypoint being wrong. Keypoints that were not hand-marked (stored as (0, 0) by
  primary_dataset) are left out entirely, since there is nothing to compare against.
- Median error in pixels, over images where a person was found, using each image's own width and height.

Results are printed and figures are saved to RESULTS_DIRECTORY.

Run every experiment with: python keypoint_detection_evaluation.py
"""
import os
from collections import defaultdict

import cv2
import keras
import matplotlib
import numpy as np

import main
import primary_dataset
from keypoint_classification import KeypointClassifier
from util import macro_f1
from pose_estimators import BlazePoseEstimator, YoloPoseEstimator, run_estimator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (the backend must be chosen before pyplot is imported)

RESULTS_DIRECTORY = "results/keypoint_detection"

_PCK_THRESHOLDS = (0.05, 0.1)

# A pipeline must run at 10 frames per second or more to be practical (Project Requirements and Specifications).
_TARGET_MILLISECONDS = 100

_SEED = 0

# Keypoints that move during a signal. Accuracy on these matters most to the classifier.
_ARM_KEYPOINTS = ["left-elbow", "left-hand", "right-elbow", "right-hand"]

# Chart colours: one fixed colour per estimator family, plus text and grid colours.
_BLAZEPOSE_COLOUR = "#2a78d6"
_YOLO_COLOUR = "#eb6834"
_TEXT_COLOUR = "#3d3d3a"
_GRID_COLOUR = "#e4e3dd"


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------

def load_images(image_paths):
    return [cv2.imread(str(image_path)) for image_path in image_paths]


def _salt_and_pepper_pixel_count(image):
    """
    Counts isolated pixels that differ sharply from their neighbours, which is the signature of salt-and-pepper noise.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    difference = np.abs(grey.astype(np.int16) - cv2.medianBlur(grey, 3).astype(np.int16))

    return int(np.sum(difference > 100))


def split_clean_and_noisy(dataset, images):
    """
    Most frames were exported twice, once without and once with salt-and-pepper noise, and both copies share a frame
    name. Returns (clean, pairs):

    clean = [index of the cleaner copy of every frame], so each frame is evaluated once
    pairs = [(clean index, noisy index)] for frames that have both copies
    """
    frame_to_indices = defaultdict(list)
    for index, frame in enumerate(dataset["Frame"]):
        frame_to_indices[frame].append(index)

    clean = []
    pairs = []

    for indices in frame_to_indices.values():
        indices = sorted(indices, key=lambda index: _salt_and_pepper_pixel_count(images[index]))
        clean.append(indices[0])

        if len(indices) == 2:
            pairs.append((indices[0], indices[1]))

    return np.array(sorted(clean)), np.array(pairs)


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------

def keypoint_errors(predicted, detected, annotated, image_sizes):
    """
    Distance in pixels between each predicted and hand-marked keypoint, with shape (images, keypoints).
    image_sizes = [(width, height)] of each image, used to convert normalised coordinates back to pixels.

    Images where no person was found are infinite, so they always count as wrong.
    Keypoints that were not hand-marked are NaN, so they are left out of every metric.
    """
    pixel_scale = np.asarray(image_sizes, dtype=np.float32)[:, None, :]
    errors = np.linalg.norm((predicted - annotated) * pixel_scale, axis=2)
    errors[~detected] = np.inf
    errors[np.all(annotated == 0, axis=2)] = np.nan

    return errors


def pck(errors, umpire_heights, threshold):
    """
    Percentage of correct keypoints for each keypoint. Missed detections (infinite) count as wrong, and keypoints
    that were not hand-marked (NaN) are left out of the denominator.
    """
    is_marked = ~np.isnan(errors)
    with np.errstate(invalid="ignore"):
        correct = errors <= threshold * umpire_heights[:, None]

    return 100 * correct.sum(axis=0) / np.maximum(is_marked.sum(axis=0), 1)


def _median_error(errors, axis=None):
    """
    Median error over images where a person was found and the keypoint was hand-marked.
    """
    return np.nanmedian(np.where(np.isinf(errors), np.nan, errors), axis=axis)


def _image_sizes(dataset, indices):
    """
    (width, height) of each image. Every image in the primary dataset is 640 x 640, which is the fallback.
    """
    if "Size" in dataset:
        return dataset["Size"][indices]

    return np.full((len(indices), 2), 640)


def summarise(name, predicted, detected, milliseconds, dataset, indices):
    """
    Accuracy and speed of one estimator over the images in indices.
    """
    errors = keypoint_errors(predicted[indices], detected[indices], dataset["X"][indices],
                             _image_sizes(dataset, indices))
    umpire_heights = dataset["BBox"][indices, 3]
    arms = [primary_dataset.BODY_KEYPOINTS.index(label) for label in _ARM_KEYPOINTS]

    summary = {
        "Estimator": name,
        "Detected (%)": 100 * detected[indices].mean(),
        "Median error (px)": float(_median_error(errors)),
        "Arm median error (px)": float(_median_error(errors[:, arms])),
        "Per keypoint PCK@0.1": pck(errors, umpire_heights, 0.1),
        "Per keypoint median error": _median_error(errors, axis=0),
        # The first image includes the time to warm up the model, so the median is used rather than the mean.
        "Time (ms)": float(np.median(milliseconds)),
    }

    for threshold in _PCK_THRESHOLDS:
        summary[f"PCK@{threshold}"] = float(pck(errors, umpire_heights, threshold).mean())
        summary[f"Arm PCK@{threshold}"] = float(pck(errors[:, arms], umpire_heights, threshold).mean())

    summary["FPS"] = 1000 / summary["Time (ms)"]

    return summary


def per_umpire_pck(predicted, detected, dataset, indices, threshold=0.1):
    """
    PCK for each umpire, to check whether accuracy holds for umpires, grounds and cameras other than the majority one.
    """
    results = {}

    for umpire in np.unique(dataset["Umpire"][indices]):
        umpire_indices = indices[dataset["Umpire"][indices] == umpire]
        errors = keypoint_errors(predicted[umpire_indices], detected[umpire_indices], dataset["X"][umpire_indices],
                                 _image_sizes(dataset, umpire_indices))
        results[umpire] = (len(umpire_indices), float(pck(errors, dataset["BBox"][umpire_indices, 3], threshold).mean()))

    return results


# ---------------------------------------------------------------------------------------------------------------------
# Image degradations and the filters used to correct them
# ---------------------------------------------------------------------------------------------------------------------

def add_salt_and_pepper(image, fraction=0.05, seed=_SEED):
    """
    Sets a random fraction of pixels to black or white, like dead or hot pixels and transmission errors.
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    mask = rng.random(image.shape[:2])
    noisy[mask < fraction / 2] = 0
    noisy[mask > 1 - fraction / 2] = 255

    return noisy


def median_filter(image, size=3):
    """
    Replaces each pixel with the median of its neighbourhood. Removes isolated outliers while keeping edges sharp.
    """
    return cv2.medianBlur(image, size)


def darken(image, factor=0.25):
    """
    Simulates an under-exposed camera, e.g. late in the day or under cloud.
    """
    return (image.astype(np.float32) * factor).astype(np.uint8)


def equalise_contrast(image):
    """
    Contrast-limited adaptive histogram equalisation (CLAHE) on the lightness channel, which spreads the narrow range
    of brightness in a dark image over the full range without changing its colours.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blur(image, sigma=4):
    """
    Gaussian blur, simulating an out-of-focus camera or motion of a fast-moving arm.
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)


def sharpen(image, sigma=4, amount=1.5):
    """
    Unsharp masking: adds back the difference between the image and a blurred copy, which strengthens edges.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def reduce_resolution(image, scale=0.25):
    """
    Downsamples then upsamples the image, simulating a distant umpire or a low-resolution phone camera.
    """
    height, width = image.shape[:2]
    small = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# (Condition, degradation, correcting filter or None)
IMAGE_QUALITY_CONDITIONS = [
    ("Salt-and-pepper noise (5%)", add_salt_and_pepper, median_filter),
    ("Low light (25% brightness)", darken, equalise_contrast),
    ("Blur (sigma = 4 px)", blur, sharpen),
    ("Low resolution (1/4 size)", reduce_resolution, None),
]


# ---------------------------------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------------------------------

def _make_estimators():
    return [BlazePoseEstimator(), YoloPoseEstimator("n"), YoloPoseEstimator("s"), YoloPoseEstimator("m")]


def compare_estimators(dataset, images, clean):
    """
    Experiment 1. Runs every estimator on every image (both copies of each frame, so the predictions can be reused by
    the classification experiment), and summarises accuracy on one copy of each frame.

    Returns {estimator name: (X, detected, milliseconds)} and the summaries.
    """
    predictions = {}
    summaries = []

    for estimator in _make_estimators():
        print(f"Running {estimator.name} on {len(images)} images...")

        with estimator:
            predictions[estimator.name] = run_estimator(estimator, images)

        summaries.append(summarise(estimator.name, *predictions[estimator.name], dataset, clean))

    return predictions, summaries


def compare_cropping(dataset, images, clean, predictions, estimator):
    """
    Experiment 2 (oracle cropping). Runs an estimator on crops around the hand-marked umpire bounding box and compares
    them with the full frame. A perfect box is the best case of what the umpire detection stage could provide, so this
    shows the most that cropping could help; it is not a test of the full pipeline with a real umpire detector.
    """
    clean_images = [images[index] for index in clean]

    with estimator:
        cropped = run_estimator(estimator, clean_images, crop_boxes=dataset["BBox"][clean])

    full_frame = [values[clean] for values in predictions[estimator.name]]
    everything = np.arange(len(clean))
    clean_dataset = {key: value[clean] for key, value in dataset.items() if key != "Classes"}

    return [
        summarise(f"{estimator.name}, full frame", *full_frame, clean_dataset, everything),
        summarise(f"{estimator.name}, oracle crop", *cropped, clean_dataset, everything),
    ]


def compare_image_quality(dataset, images, clean, predictions, estimators):
    """
    Experiment 3. For each degradation, runs the estimators on the degraded images, and on the degraded images after
    the correcting filter. Returns {estimator name: [(condition, PCK@0.1 degraded, PCK@0.1 corrected or None)]}.
    """
    clean_images = [images[index] for index in clean]
    clean_dataset = {key: value[clean] for key, value in dataset.items() if key != "Classes"}
    everything = np.arange(len(clean))
    results = {}

    for estimator in estimators:
        baseline = summarise(estimator.name, *[values[clean] for values in predictions[estimator.name]],
                             clean_dataset, everything)
        rows = [("Original", baseline["PCK@0.1"], None)]

        with estimator:
            for condition, degrade, correct in IMAGE_QUALITY_CONDITIONS:
                print(f"{estimator.name}: {condition}...")
                degraded_images = [degrade(image) for image in clean_images]
                degraded = summarise(estimator.name, *run_estimator(estimator, degraded_images),
                                     clean_dataset, everything)

                corrected = None
                if correct is not None:
                    corrected_images = [correct(image) for image in degraded_images]
                    corrected = summarise(estimator.name, *run_estimator(estimator, corrected_images),
                                          clean_dataset, everything)["PCK@0.1"]

                rows.append((condition, degraded["PCK@0.1"], corrected))

        results[estimator.name] = rows

    return results


def compare_dataset_noise(dataset, pairs, predictions):
    """
    Experiment 4. Accuracy on the clean and noisy copies of the same frames, using the predictions from experiment 1.
    """
    rows = []

    for name, values in predictions.items():
        clean_summary = summarise(name, *values, dataset, pairs[:, 0])
        noisy_summary = summarise(name, *values, dataset, pairs[:, 1])
        rows.append((name, clean_summary["PCK@0.1"], noisy_summary["PCK@0.1"]))

    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------------

def _print_table(rows, columns):
    widths = [max(len(column), 10) for column in columns]
    print("  ".join(column.rjust(width) for column, width in zip(columns, widths)))

    for row in rows:
        cells = [f"{row[column]:.1f}" if isinstance(row[column], float) else str(row[column]) for column in columns]
        print("  ".join(cell.rjust(width) for cell, width in zip(cells, widths)))


_SUMMARY_COLUMNS = ["Estimator", "Detected (%)", "PCK@0.05", "PCK@0.1", "Arm PCK@0.1", "Median error (px)",
                    "Arm median error (px)", "Time (ms)", "FPS"]


def _style_axes(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(_GRID_COLOUR)
    axes.tick_params(colors=_TEXT_COLOUR, length=0)
    axes.yaxis.grid(True, color=_GRID_COLOUR, linewidth=0.8)
    axes.set_axisbelow(True)


def _colour(name):
    return _BLAZEPOSE_COLOUR if name.startswith("BlazePose") else _YOLO_COLOUR


def plot_example(dataset, images, index, predictions, names, path):
    """
    Figure: one image with the hand-marked keypoints and each estimator's keypoints overlaid.
    """
    figure, all_axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.4))

    for axes, name in zip(all_axes, names):
        axes.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        height, width = images[index].shape[:2]
        annotated = dataset["X"][index] * [width, height]
        predicted = predictions[name][0][index] * [width, height]
        axes.scatter(*annotated.T, s=60, facecolors="none", edgecolors="white", linewidths=2, label="Hand-marked")
        axes.scatter(*predicted.T, s=22, color=_colour(name), edgecolors="white", linewidths=0.8, label=name)
        axes.set_title(name, color=_TEXT_COLOUR, fontsize=11)
        axes.axis("off")
        axes.legend(loc="lower center", fontsize=8, frameon=True)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_per_keypoint(summaries, path):
    """
    Figure: PCK@0.1 for each keypoint, for each estimator, as horizontal grouped bars.
    """
    labels = primary_dataset.BODY_KEYPOINTS
    order = np.argsort(summaries[0]["Per keypoint PCK@0.1"])
    bar_height = 0.8 / len(summaries)

    figure, axes = plt.subplots(figsize=(7, 6.5))

    for i, summary in enumerate(summaries):
        positions = np.arange(len(labels)) + (i - (len(summaries) - 1) / 2) * bar_height
        axes.barh(positions, summary["Per keypoint PCK@0.1"][order], height=bar_height - 0.03,
                  color=_colour(summary["Estimator"]), label=summary["Estimator"])

    axes.set_yticks(np.arange(len(labels)), [labels[i] for i in order])
    axes.set_xlabel("PCK@0.1 (%)", color=_TEXT_COLOUR)
    axes.set_xlim(0, 100)
    _style_axes(axes)
    axes.yaxis.grid(False)
    axes.xaxis.grid(True, color=_GRID_COLOUR, linewidth=0.8)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(summaries), frameon=False)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_accuracy_against_speed(summaries, path):
    """
    Figure: accuracy (PCK@0.1) against time per image, with the 10 frames per second limit marked.
    """
    figure, axes = plt.subplots(figsize=(6.5, 4.2))

    for summary in summaries:
        axes.scatter(summary["Time (ms)"], summary["PCK@0.1"], s=80, color=_colour(summary["Estimator"]),
                     edgecolors="white", linewidths=2, zorder=3)
        axes.annotate(summary["Estimator"], (summary["Time (ms)"], summary["PCK@0.1"]),
                      textcoords="offset points", xytext=(8, 6), fontsize=9, color=_TEXT_COLOUR)

    slowest = max(summary["Time (ms)"] for summary in summaries)
    axes.set_xlim(0, slowest * 1.25)
    axes.axvline(_TARGET_MILLISECONDS, color=_TEXT_COLOUR, linestyle="--", linewidth=1)
    axes.annotate("10 FPS limit", (_TARGET_MILLISECONDS, axes.get_ylim()[1]), textcoords="offset points",
                  xytext=(5, -14), fontsize=9, color=_TEXT_COLOUR)
    axes.set_xlabel("Time per image on CPU (ms)", color=_TEXT_COLOUR)
    axes.set_ylabel("PCK@0.1 (%)", color=_TEXT_COLOUR)
    _style_axes(axes)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_image_quality(results, path):
    """
    Figure: PCK@0.1 under each degradation, before and after the correcting filter, one panel per estimator.
    """
    figure, all_axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5), sharey=True)

    for axes, (name, rows) in zip(np.atleast_1d(all_axes), results.items()):
        positions = np.arange(len(rows))
        degraded = [row[1] for row in rows]
        corrected = [np.nan if row[2] is None else row[2] for row in rows]

        axes.bar(positions - 0.2, degraded, width=0.38, color=_colour(name), label="Degraded")
        axes.bar(positions + 0.2, corrected, width=0.38, color=_colour(name), alpha=0.45, label="After filter")

        for position, value in zip(positions, degraded):
            axes.annotate(f"{value:.0f}", (position - 0.2, value), textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=8, color=_TEXT_COLOUR)
        for position, value in zip(positions, corrected):
            if not np.isnan(value):
                axes.annotate(f"{value:.0f}", (position + 0.2, value), textcoords="offset points", xytext=(0, 3),
                              ha="center", fontsize=8, color=_TEXT_COLOUR)

        axes.set_xticks(positions, [row[0].replace(" (", "\n(") for row in rows], fontsize=8)
        axes.set_title(name, color=_TEXT_COLOUR, pad=30)
        axes.set_ylim(0, 105)
        _style_axes(axes)
        axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)

    np.atleast_1d(all_axes)[0].set_ylabel("PCK@0.1 (%)", color=_TEXT_COLOUR)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------------

def run_all():
    """
    Runs every experiment, prints the results and saves the figures.
    Returns the dataset and {estimator name: (X, detected, milliseconds)} on every image, for the classification
    experiment in main.py, along with the name of the most accurate YOLO model.
    """
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    dataset = primary_dataset.load_all_annotations()
    images = load_images(dataset["Image"])
    dataset["Size"] = np.array([(image.shape[1], image.shape[0]) for image in images])
    clean, pairs = split_clean_and_noisy(dataset, images)
    print(f"{len(images)} images, {len(clean)} unique frames, {len(pairs)} frames with a noisy copy")

    print("\n\n--- 1. ESTIMATOR COMPARISON (one copy of each frame) ---\n")
    predictions, summaries = compare_estimators(dataset, images, clean)
    _print_table(summaries, _SUMMARY_COLUMNS)

    print("\nPCK@0.1 per umpire:")
    for summary in summaries:
        values = per_umpire_pck(*predictions[summary["Estimator"]][:2], dataset, clean)
        print(f"{summary['Estimator']:>18}: " + ", ".join(f"{u} ({n}) {v:.1f}" for u, (n, v) in values.items()))

    print("\nMedian error per keypoint (px):")
    for summary in summaries:
        print(f"{summary['Estimator']:>18}: " + ", ".join(
            f"{label} {error:.1f}"
            for label, error in zip(primary_dataset.BODY_KEYPOINTS, summary["Per keypoint median error"])
        ))

    yolo_summaries = [summary for summary in summaries if summary["Estimator"].startswith("YOLO")]
    best_yolo = max(yolo_summaries, key=lambda summary: summary["PCK@0.1"])
    best_yolo_size = best_yolo["Estimator"][len("YOLO11")]
    print(f"\nMost accurate YOLO model: {best_yolo['Estimator']}")

    print("\n\n--- 2. FULL FRAME VS ORACLE CROP (HAND-MARKED UMPIRE BOX) ---\n")
    cropping = (compare_cropping(dataset, images, clean, predictions, BlazePoseEstimator())
                + compare_cropping(dataset, images, clean, predictions, YoloPoseEstimator(best_yolo_size)))
    _print_table(cropping, _SUMMARY_COLUMNS)

    print("\n\n--- 3. IMAGE QUALITY (PCK@0.1) ---\n")
    image_quality = compare_image_quality(dataset, images, clean, predictions,
                                          [BlazePoseEstimator(), YoloPoseEstimator(best_yolo_size)])
    for name, rows in image_quality.items():
        print(name)
        for condition, degraded, corrected in rows:
            after = "" if corrected is None else f" -> {corrected:.1f} after filter"
            print(f"  {condition:>28}: {degraded:.1f}{after}")

    print("\n\n--- 4. DATASET NOISE: CLEAN VS SALT-AND-PEPPER COPY (PCK@0.1) ---\n")
    for name, clean_pck, noisy_pck in compare_dataset_noise(dataset, pairs, predictions):
        print(f"{name:>18}: clean {clean_pck:.1f}, noisy {noisy_pck:.1f}")

    example = clean[np.argmax(dataset["Umpire"][clean] == "imgE")]
    plot_example(dataset, images, example, predictions, ["BlazePose (full)", best_yolo["Estimator"]],
                 f"{RESULTS_DIRECTORY}/example_keypoints.png")
    plot_per_keypoint([summaries[0], best_yolo], f"{RESULTS_DIRECTORY}/pck_per_keypoint.png")
    plot_accuracy_against_speed(summaries, f"{RESULTS_DIRECTORY}/accuracy_against_speed.png")
    plot_image_quality(image_quality, f"{RESULTS_DIRECTORY}/image_quality.png")
    print(f"\nFigures saved to {RESULTS_DIRECTORY}")

    return dataset, predictions, best_yolo["Estimator"]


def evaluate_classification_on_unseen_umpires(dataset, x, detected=None, seed=_SEED):
    """
    Trains and tests the keypoint classifier on keypoints x with leave-one-umpire-out, in the same way as
    main.test_keypoint_classification_on_unseen_umpires, and returns (accuracy, macro F1) over all umpires.
    Images where no person was detected are left out of training and count as wrong when testing.
    """
    source = dict(dataset, X=x)
    if detected is None:
        detected = np.ones(len(source["Y"]), dtype=bool)

    num_classes = len(source["Classes"])
    all_true = []
    all_predicted = []

    for umpire in np.unique(source["Umpire"]):
        keras.utils.set_random_seed(seed)

        is_test = source["Umpire"] == umpire
        train, validate = main._split_validate(main._subset(source, ~is_test & detected))
        test = main._subset(source, is_test)

        keypoint_classifier = KeypointClassifier(train["X"].shape[1], num_classes)
        keypoint_classifier.fit(train, validate, verbose=0)

        # A missed detection is a failure of the whole pipeline, so it is never a correct prediction.
        all_predicted.append(np.where(detected[is_test], keypoint_classifier.predict(test["X"]), -1))
        all_true.append(test["Y"])

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)

    return np.mean(all_predicted == all_true), macro_f1(all_true, all_predicted, num_classes)


def compare_classification(dataset, predictions, best_yolo, num_seeds=3):
    """
    Experiment 5. For each source of keypoints, the keypoint classifier is trained and tested on keypoints from that
    source with leave-one-umpire-out, repeated with several random seeds. Hand-marked keypoints give the upper bound,
    i.e. the accuracy the classifier would reach with a perfect keypoint detector.
    """
    sources = [("Hand-marked keypoints", dataset["X"], None)]
    for name in ["BlazePose (full)", best_yolo]:
        predicted_x, detected, _ = predictions[name]
        sources.append((name, predicted_x, detected))

    print("\n\n--- 5. SIGNAL CLASSIFICATION ON UNSEEN UMPIRES, BY KEYPOINT SOURCE ---\n")
    print(f"{'Keypoints':>22}  {'Macro F1':>13}  {'Accuracy':>13}")

    for name, x, detected in sources:
        scores = [evaluate_classification_on_unseen_umpires(dataset, x, detected, seed) for seed in range(num_seeds)]
        means = np.mean(scores, axis=0)
        sds = np.std(scores, axis=0)
        print(f"{name:>22}  {means[1]:.3f} ± {sds[1]:.3f}  {means[0]:.3f} ± {sds[0]:.3f}")


if __name__ == "__main__":
    compare_classification(*run_all())
