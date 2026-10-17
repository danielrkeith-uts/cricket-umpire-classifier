import json
from collections import defaultdict

import numpy as np

_TRAIN_PATH = "datasets/primary/train"
_TEST_PATH = "datasets/primary/test"
_VALIDATE_PATH = "datasets/primary/valid"
_ANNOTATIONS_FILE = "_annotations.coco.json"

CLASSES = ["Out", "Four", "Six", "Wide", "No-ball", "Idle"]

# The set of annotation categories on an image determines its class.
# Out and No-ball images have a body annotation plus a separate hand annotation.
# Images with any other set of categories (bye, leg-bye, dead-ball, etc.) are dropped.
_CATEGORIES_TO_CLASS = {
    frozenset({"hand up", "out"}): "Out",
    frozenset({"4"}): "Four",
    frozenset({"6"}): "Six",
    frozenset({"wide"}): "Wide",
    frozenset({"hand out", "no ball"}): "No-ball",
    frozenset({"idle"}): "Idle",
}

# Categories annotated with hand keypoints only. These decide the class but are not used as features,
# since only Out and No-ball images have them.
_HAND_CATEGORIES = {"out", "no ball", "bye"}

_BODY_KEYPOINTS = [
    "body",
    "head",
    "left-elbow",
    "left-foot",
    "left-hand",
    "left-hip",
    "left-knee",
    "left-shoulder",
    "neck",
    "right-elbow",
    "right-foot",
    "right-hand",
    "right-hip",
    "right-knee",
    "right-shoulder",
]

_BODY_KEYPOINT_TO_INDEX = {
    label: index
    for index, label in enumerate(_BODY_KEYPOINTS)
}


def _reshape_coco_annotations(coco_annotations):
    """
    Takes coco dataset and returns it in the form (X, Y) where, for each image:

    X = [array of every body keypoint [(x1, y1), (x2, y2), (x3, y3)], each normalised to range [0,1]]

    Y = [index into CLASSES]

    Images whose annotations do not correspond to one of CLASSES are dropped.
    """

    id_to_category = {
        category["id"]: category
        for category in coco_annotations["categories"]
    }

    image_id_to_annotations = defaultdict(list)
    for annotation in coco_annotations["annotations"]:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    x = []
    y = []

    for image in coco_annotations["images"]:
        annotations = image_id_to_annotations[image["id"]]

        category_names = frozenset(id_to_category[annotation["category_id"]]["name"] for annotation in annotations)
        class_name = _CATEGORIES_TO_CLASS.get(category_names)

        if class_name is None:
            continue

        body_annotations = [
            annotation
            for annotation in annotations
            if id_to_category[annotation["category_id"]]["name"] not in _HAND_CATEGORIES
        ]

        # A few images have a second, tiny body annotation in the corner of the image. Use the largest.
        body_annotation = max(body_annotations, key=lambda annotation: annotation["bbox"][2] * annotation["bbox"][3])

        x.append(_extract_body_keypoints(body_annotation, id_to_category[body_annotation["category_id"]], image))
        y.append(CLASSES.index(class_name))

    return {
        "X": np.array(x),
        "Y": np.array(y),
        "Classes": dict(enumerate(CLASSES))
    }


def _extract_body_keypoints(annotation, category, image):
    # An x- and y-feature for each keypoint.
    # Leave unknown values as zero, since a (0, 0) coordinate will not appear in real data anyway.
    features = np.zeros((len(_BODY_KEYPOINTS), 2), dtype=np.float32)

    category_keypoint_labels = category.get("keypoints", [])

    raw_provided_keypoints = np.asarray(
        annotation.get("keypoints", []),
        dtype=np.float32
    )

    # The raw provided keypoints include the x-value, y-value, and another unknown value.
    # This last value will be discarded; i.e., every third element will be filtered out.
    provided_keypoints_mask = (np.arange(len(raw_provided_keypoints)) + 1) % 3 != 0
    provided_keypoints = raw_provided_keypoints[provided_keypoints_mask]

    if len(provided_keypoints) != len(category_keypoint_labels) * 2:
        raise ValueError(
            f"Annotation {annotation['id']}: "
            f"expected {len(category_keypoint_labels) * 2} keypoints, "
            f"got {len(provided_keypoints)} keypoints"
        )

    # Reshape so that each keypoint contains the x- and y- value.
    provided_keypoints = provided_keypoints.reshape(-1, 2)

    # Get image width and height for normalisation
    width = image["width"]
    height = image["height"]

    for label, (x_coord, y_coord) in zip(category_keypoint_labels, provided_keypoints):
        feature_index = _BODY_KEYPOINT_TO_INDEX[label]

        features[feature_index] = (x_coord / width, y_coord / height)

    return features


def _load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def _load_and_reshape_annotations(path):
    return _reshape_coco_annotations(_load_json(path))


def load_train_annotations():
    return _load_and_reshape_annotations(f"{_TRAIN_PATH}/{_ANNOTATIONS_FILE}")


def load_test_annotations():
    return _load_and_reshape_annotations(f"{_TEST_PATH}/{_ANNOTATIONS_FILE}")


def load_validate_annotations():
    return _load_and_reshape_annotations(f"{_VALIDATE_PATH}/{_ANNOTATIONS_FILE}")
