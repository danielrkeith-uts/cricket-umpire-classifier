import json

def standardise_coco_data(coco_data):
    keypoint_labels = sorted(
        label
        for category in coco_data["categories"]
        for label in category.get("keypoints", [])
    )

    keypoint_label_to_feature_index = {
        label: feature_index
        for feature_index, label in enumerate(keypoint_labels)
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
        category = id_to_category[annotation["category_id"]]
        category_keypoints = category.get("keypoints", [])

        pass

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
