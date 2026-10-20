import primary_dataset, util

import time

import numpy as np
import tensorflow as tf

from keras.src.callbacks import EarlyStopping
from tensorflow.keras import layers, models

_IMAGE_SIZE = 224

_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


def _load_images(dataset):
    """
    Loads each image resized to _IMAGE_SIZE x _IMAGE_SIZE, and returns (images, boxes) where:
    images = [pixel values, as uint8 to save memory]
    boxes = [umpire bounding box (x, y, width, height)], normalised to range [0, 1] using the original image size
    """
    images = np.zeros((len(dataset["Image"]), _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
    boxes = np.zeros((len(dataset["Image"]), 4), dtype=np.float32)

    for i, (image_path, (x, y, width, height)) in enumerate(zip(dataset["Image"], dataset["BBox"])):
        image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
        image_height, image_width = image.shape[0], image.shape[1]
        images[i] = tf.image.resize(image, (_IMAGE_SIZE, _IMAGE_SIZE)).numpy().astype(np.uint8)
        boxes[i] = (x / image_width, y / image_height, width / image_width, height / image_height)

    return images, boxes


def _box_iou(box_a, box_b):
    """
    Intersection over union of boxes (x, y, width, height), compared pairwise.
    """
    left = np.maximum(box_a[:, 0], box_b[:, 0])
    top = np.maximum(box_a[:, 1], box_b[:, 1])
    right = np.minimum(box_a[:, 0] + box_a[:, 2], box_b[:, 0] + box_b[:, 2])
    bottom = np.minimum(box_a[:, 1] + box_a[:, 3], box_b[:, 1] + box_b[:, 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = box_a[:, 2] * box_a[:, 3] + box_b[:, 2] * box_b[:, 3] - intersection
    return intersection / np.maximum(union, 1e-9)


class SignalObjectDetector:
    """
    Single-object detector: since there is one umpire per image, the model predicts one bounding box (where the
    umpire is) and one class (which signal they are making) directly from the image pixels.
    """

    def __init__(self, num_classes):
        # MobileNetV2 pretrained on ImageNet extracts visual features. It is frozen, since the dataset is too small
        # to retrain it without overfitting.
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        backbone.trainable = False

        inputs = layers.Input(shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3))
        # Scale pixels from [0, 255] to [-1, 1], the range MobileNetV2 was trained on.
        features = backbone(layers.Rescaling(1 / 127.5, offset=-1)(inputs))
        features = layers.Dense(128, activation="relu")(features)
        features = layers.Dropout(0.2)(features)

        # Two outputs: the signal class, and the box (x, y, width, height) normalised to range [0, 1].
        signal = layers.Dense(num_classes, activation="softmax", name="signal")(features)
        box = layers.Dense(4, activation="sigmoid", name="box")(features)

        self.model = models.Model(inputs, [signal, box])
        self.model.compile(
            optimizer="adam",
            loss={"signal": "sparse_categorical_crossentropy", "box": "mse"},
            metrics={"signal": "accuracy"},
        )

    def fit(self, train, validate, verbose="auto"):
        self.model.fit(
            train["Images"],
            {"signal": train["Y"], "box": train["Boxes"]},
            epochs=200,
            batch_size=32,
            callbacks=[_early_stopping],
            validation_data=(validate["Images"], {"signal": validate["Y"], "box": validate["Boxes"]}),
            verbose=verbose
        )

    def predict(self, images):
        """
        Returns (Y, Boxes): the predicted class index and normalised box for each image.
        """
        probabilities, boxes = self.model.predict(images, verbose=0)
        return np.argmax(probabilities, axis=1), boxes


def _prepare(dataset):
    images, boxes = _load_images(dataset)
    return {"Images": images, "Boxes": boxes, "Y": dataset["Y"]}


def _subset(dataset, mask):
    return {key: value[mask] for key, value in dataset.items()}


def _print_results(name, y_true, predicted, iou, milliseconds=None):
    four = primary_dataset.CLASSES.index("Four")
    no_ball = primary_dataset.CLASSES.index("No-ball")

    print(f"\n\n--- OBJECT DETECTION: {name.upper()} ---\n")
    print(
        f"Accuracy {np.mean(predicted == y_true):.3f}, macro F1 {util.macro_f1(y_true, predicted, len(primary_dataset.CLASSES)):.3f}")
    print(f"Mean IoU {np.mean(iou):.3f}, umpire located (IoU >= 0.5) in {np.mean(iou >= 0.5):.3f} of images")
    print(f"Four classified as No-ball {np.mean(predicted[y_true == four] == no_ball):.3f}")
    if milliseconds is not None:
        print(f"{milliseconds:.1f} ms per image ({1000 / milliseconds:.1f} FPS)")


def test_object_detection():
    """
    1. Trains on the dataset's train split (validate split for early stopping) and tests on the test split.
    2. Leave-one-umpire-out: for each umpire, trains on every other umpire and tests on that one, the same
       procedure used for the keypoint classifier and the LSTM, so the results can be compared directly.
    """
    print("\nLoading images...")
    train = _prepare(primary_dataset.load_train_annotations())
    validate = _prepare(primary_dataset.load_validate_annotations())
    test = _prepare(primary_dataset.load_test_annotations())

    detector = SignalObjectDetector(len(primary_dataset.CLASSES))
    detector.fit(train, validate, verbose=0)

    # Run once before timing, so the one-off setup cost is not counted in the speed.
    detector.predict(test["Images"][:1])
    start = time.perf_counter()
    predicted, boxes = detector.predict(test["Images"])
    milliseconds = (time.perf_counter() - start) * 1000 / len(test["Y"])
    _print_results("Dataset test split", test["Y"], predicted, _box_iou(boxes, test["Boxes"]), milliseconds)

    # Leave-one-umpire-out. Both copies of each frame (with and without noise) stay on the same side of the
    # train/validate split, so validation images are not near-duplicates of training images.
    dataset = primary_dataset.load_all_annotations()
    everything = dict(_prepare(dataset), Umpire=dataset["Umpire"], Frame=dataset["Frame"])
    rng = np.random.default_rng(0)
    all_true, all_predicted, all_iou = [], [], []

    for umpire in np.unique(everything["Umpire"]):
        is_test = everything["Umpire"] == umpire
        rest = _subset(everything, ~is_test)
        frames = np.unique(rest["Frame"])
        is_validate = np.isin(rest["Frame"], rng.choice(frames, size=round(len(frames) * 0.15), replace=False))

        detector = SignalObjectDetector(len(primary_dataset.CLASSES))
        detector.fit(_subset(rest, ~is_validate), _subset(rest, is_validate), verbose=0)

        umpire_test = _subset(everything, is_test)
        predicted, boxes = detector.predict(umpire_test["Images"])
        print(f"{umpire:>10}: accuracy {np.mean(predicted == umpire_test['Y']):.3f}")
        all_true.append(umpire_test["Y"])
        all_predicted.append(predicted)
        all_iou.append(_box_iou(boxes, umpire_test["Boxes"]))

    _print_results("Unseen umpires", np.concatenate(all_true), np.concatenate(all_predicted),
                   np.concatenate(all_iou))
