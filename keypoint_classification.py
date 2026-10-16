import json
import numpy as np


def standardise_coco_data(coco_data):
    keypoint_labels = sorted({
        label
        for category in coco_data["categories"]
        for label in category.get("keypoints", [])
    })

    keypoint_label_to_index = {
        label: index
        for index, label in enumerate(keypoint_labels)
    }

    id_to_category = {
        category["id"]: category
        for category in coco_data["categories"]
    }

    id_to_image = {
        image["id"]: image
        for image in coco_data["images"]
    }

    x = []
    y = []

    for annotation in coco_data["annotations"]:
        # An x- and y-feature for each keypoint.
        # Every keypoint not specified will remain zero.
        features = np.full(len(keypoint_labels) * 2, np.nan, dtype=np.float32)

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

        for label, (x_coord, y_coord) in zip(category_keypoint_labels, provided_keypoints):
            feature_index = keypoint_label_to_index[label] * 2

            features[feature_index] = x_coord
            features[feature_index + 1] = y_coord

        x.append(features)
        y.append(category["name"])

    return x, y


class KeypointClassifier:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def load_training_data_from_coco(self, path):
        with open(path) as json_file:
            raw_training_data = json.load(json_file)

        (self.x_train, self.y_train) = standardise_coco_data(raw_training_data)

        pass
