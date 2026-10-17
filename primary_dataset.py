import json

import numpy as np

_TRAIN_PATH = "datasets/primary/train"
_TEST_PATH = "datasets/primary/test"
_VALIDATE_PATH = "datasets/primary/valid"
_ANNOTATIONS_FILE = "_annotations.coco.json"


def _reshape_coco_annotations(coco_annotations):
    """
    Takes coco dataset and returns it in the form (X, Y) where:

    X = [array of every keypoint [(x1, y1), (x2, y2), (x3, y3)], each normalised to range [0,1]]

    Y = [class label]

    Note that the dataset does not include every keypoint on every image, so there are many missing values.
    """

    keypoint_labels = sorted({
        label
        for category in coco_annotations["categories"]
        for label in category.get("keypoints", [])
    })

    keypoint_label_to_index = {
        label: index
        for index, label in enumerate(keypoint_labels)
    }

    id_to_category = {
        category["id"]: category
        for category in coco_annotations["categories"]
    }

    id_to_image = {
        image["id"]: image
        for image in coco_annotations["images"]
    }

    x = []
    y = []

    for annotation in coco_annotations["annotations"]:
        # An x- and y-feature for each keypoint.
        structured_type = [('x', 'float'), ('y', 'float')]
        features = np.full(len(keypoint_labels), np.nan, dtype=structured_type)

        category = id_to_category[annotation["category_id"]]
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
        image = id_to_image[annotation["image_id"]]
        width = image["width"]
        height = image["height"]

        for label, (x_coord, y_coord) in zip(category_keypoint_labels, provided_keypoints):
            feature_index = keypoint_label_to_index[label]

            features[feature_index] = (x_coord / width, y_coord / height)

        x.append(features)
        y.append(category["name"])

    return x, y


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
