from tensorflow.keras import models, layers

class KeypointClassifier:
    def __init__(self, num_keypoints, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=(num_keypoints, 2)),
            layers.Dense(num_classes, activation="softmax")
        ])
        pass


    def fit(self, train, validate):
        pass
