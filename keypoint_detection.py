import os
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from primary_dataset import BODY_KEYPOINTS

_MODEL_PATH = "models/pose_landmarker_full.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# BlazePose predicts 33 landmarks (see vision.PoseLandmark). Each dataset keypoint is the mean of these landmarks,
# chosen as the closest match to the annotated keypoints. Left and right are the umpire's own left and right in both.
_BODY_KEYPOINT_TO_LANDMARKS = {
    "body": [11, 12, 23, 24],  # Centre of the shoulders and hips
    "head": [2, 5],  # Eyes
    "left-elbow": [13],
    "left-foot": [31],  # Toe
    "left-hand": [15, 17, 19, 21],  # Wrist, pinky, index and thumb
    "left-hip": [23],
    "left-knee": [25],
    "left-shoulder": [11],
    "neck": [9, 10, 11, 11, 12, 12],  # Two thirds of the way from the mouth to the shoulders
    "right-elbow": [14],
    "right-foot": [32],
    "right-hand": [16, 18, 20, 22],
    "right-hip": [24],
    "right-knee": [26],
    "right-shoulder": [12],
}


def _ensure_model_downloaded():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def detect_body_keypoints(image_paths):
    """
    Runs BlazePose on each image and returns (X, detected) where:

    X = [array of every body keypoint [(x1, y1), (x2, y2), (x3, y3)], each normalised to range [0,1]],
        in the same layout as primary_dataset

    detected = [whether a person was found]; images with no person have all keypoints left as zero.
    """
    _ensure_model_downloaded()

    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=_MODEL_PATH))

    x = np.zeros((len(image_paths), len(BODY_KEYPOINTS), 2), dtype=np.float32)
    detected = np.zeros(len(image_paths), dtype=bool)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, image_path in enumerate(image_paths):
            result = landmarker.detect(mp.Image.create_from_file(str(image_path)))

            if not result.pose_landmarks:
                continue

            # Landmarks are already normalised to the image width and height.
            landmarks = np.array([(landmark.x, landmark.y) for landmark in result.pose_landmarks[0]], dtype=np.float32)

            for feature_index, label in enumerate(BODY_KEYPOINTS):
                x[i, feature_index] = landmarks[_BODY_KEYPOINT_TO_LANDMARKS[label]].mean(axis=0)

            detected[i] = True

    return x, detected
