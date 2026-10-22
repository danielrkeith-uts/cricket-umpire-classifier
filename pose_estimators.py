"""
Pose estimators that turn an image of an umpire into the 15 body keypoints used by the rest of the project.

Every estimator returns keypoints in the same layout as primary_dataset (BODY_KEYPOINTS order, x and y normalised to
[0, 1] by the image width and height), so its output can be passed straight to the keypoint classifier or the LSTM.

Two families of estimator are compared:
- BlazePose (MediaPipe Pose Landmarker): a top-down model that first finds one person, then predicts 33 landmarks.
- YOLO11-pose (Ultralytics): a single-stage model that finds every person and their 17 COCO keypoints in one pass.

keypoint_detection.py (BlazePose on image files) is used by the classifier and LSTM experiments. This module works on
images that are already loaded as arrays, so that they can be cropped or degraded before keypoint detection.
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from keypoint_detection import _BODY_KEYPOINT_TO_LANDMARKS
from primary_dataset import BODY_KEYPOINTS

_MODELS_DIRECTORY = "models"

_BLAZEPOSE_MODEL_PATH = f"{_MODELS_DIRECTORY}/pose_landmarker_full.task"
_BLAZEPOSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

_YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}.pt"

# YOLO predicts the 17 COCO keypoints:
# 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
# (odd indices are the person's left, even indices their right).
# Each dataset keypoint is the mean of the closest COCO keypoints, following the same choices as the BlazePose mapping
# in keypoint_detection.py. COCO has no finger or toe keypoints, so hands are the wrists and feet are the ankles.
_BODY_KEYPOINT_TO_COCO = {
    "body": [5, 6, 11, 12],  # Centre of the shoulders and hips
    "head": [1, 2],  # Eyes
    "left-elbow": [7],
    "left-foot": [15],  # Ankle
    "left-hand": [9],  # Wrist
    "left-hip": [11],
    "left-knee": [13],
    "left-shoulder": [5],
    "neck": [0, 0, 5, 5, 5, 6, 6, 6],  # A quarter of the way from the shoulders to the nose
    "right-elbow": [8],
    "right-foot": [16],
    "right-hand": [10],
    "right-hip": [12],
    "right-knee": [14],
    "right-shoulder": [6],
}


def _download_if_missing(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def _map_to_body_keypoints(points, mapping):
    """
    Converts an estimator's own keypoints (array of normalised (x, y)) into the BODY_KEYPOINTS layout.
    """
    keypoints = np.zeros((len(BODY_KEYPOINTS), 2), dtype=np.float32)

    for feature_index, label in enumerate(BODY_KEYPOINTS):
        keypoints[feature_index] = points[mapping[label]].mean(axis=0)

    return keypoints


class BlazePoseEstimator:
    """
    BlazePose (full model). Use inside a with-block so the model is loaded once and released afterwards.
    """
    name = "BlazePose (full)"

    def __enter__(self):
        _download_if_missing(_BLAZEPOSE_MODEL_URL, _BLAZEPOSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=_BLAZEPOSE_MODEL_PATH))
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, *exc):
        self._landmarker.close()

    def detect(self, image_bgr):
        """
        Returns the 15 body keypoints of the person in a BGR image (as loaded by OpenCV), or None if nobody is found.
        """
        image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))

        if not result.pose_landmarks:
            return None

        # Landmarks are already normalised to the image width and height.
        landmarks = np.array([(landmark.x, landmark.y) for landmark in result.pose_landmarks[0]], dtype=np.float32)

        return _map_to_body_keypoints(landmarks, _BODY_KEYPOINT_TO_LANDMARKS)


class YoloPoseEstimator:
    """
    YOLO11-pose. size is "n" (nano), "s" (small) or "m" (medium); larger models are slower but more accurate.
    """

    def __init__(self, size):
        self._model_name = f"yolo11{size}-pose"
        self.name = f"YOLO11{size}-pose"

    def __enter__(self):
        # Imported here so the rest of the project does not need ultralytics installed.
        from ultralytics import YOLO

        model_path = f"{_MODELS_DIRECTORY}/{self._model_name}.pt"
        _download_if_missing(_YOLO_MODEL_URL.format(name=self._model_name), model_path)
        self._model = YOLO(model_path)
        return self

    def __exit__(self, *exc):
        pass

    def detect(self, image_bgr):
        """
        Returns the 15 body keypoints of the most confidently detected person, or None if nobody is found.
        Choosing the umpire among several people is left to the umpire detection stage.
        """
        result = self._model.predict(image_bgr, device="cpu", verbose=False)[0]

        if result.keypoints is None or len(result.boxes) == 0:
            return None

        best_person = int(np.argmax(result.boxes.conf.cpu().numpy()))
        points = result.keypoints.xyn[best_person].cpu().numpy()

        return _map_to_body_keypoints(points, _BODY_KEYPOINT_TO_COCO)


def run_estimator(estimator, images, crop_boxes=None):
    """
    Runs an estimator (already entered with a with-block) over BGR images and returns (X, detected, milliseconds):

    X = [15 body keypoints per image, normalised to the full image], all zero where nobody was found
    detected = [whether a person was found]
    milliseconds = [time taken by the estimator on each image]

    When crop_boxes ([x, y, width, height] in pixels per image) are given, each image is cropped to its box first and
    the keypoints are mapped back onto the full image. This simulates receiving the umpire from the umpire detector.
    """
    x = np.zeros((len(images), len(BODY_KEYPOINTS), 2), dtype=np.float32)
    detected = np.zeros(len(images), dtype=bool)
    milliseconds = np.zeros(len(images), dtype=np.float32)

    for i, image in enumerate(images):
        height, width = image.shape[:2]

        if crop_boxes is not None:
            left, top, right, bottom = _box_to_crop(crop_boxes[i], width, height)
            image = image[top:bottom, left:right]

        start = time.perf_counter()
        keypoints = estimator.detect(image)
        milliseconds[i] = (time.perf_counter() - start) * 1000

        if keypoints is None:
            continue

        if crop_boxes is not None:
            # Convert from coordinates normalised to the crop back to coordinates normalised to the full image.
            keypoints = (keypoints * [right - left, bottom - top] + [left, top]) / [width, height]

        x[i] = keypoints
        detected[i] = True

    return x, detected, milliseconds


# Fraction of the box's width and height added on every side, so that raised hands are not cut off.
_CROP_MARGIN = 0.2


def _box_to_crop(box, width, height):
    box_x, box_y, box_width, box_height = box
    left = int(max(0, box_x - _CROP_MARGIN * box_width))
    top = int(max(0, box_y - _CROP_MARGIN * box_height))
    right = int(min(width, box_x + (1 + _CROP_MARGIN) * box_width))
    bottom = int(min(height, box_y + (1 + _CROP_MARGIN) * box_height))

    return left, top, right, bottom
