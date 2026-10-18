import re
from collections import defaultdict

import numpy as np
from tensorflow.keras import layers, models

# Frames from the same video up to this far apart are treated as part of one continuous clip.
_MAX_FRAME_GAP = 3

# e.g. "imgD_233.jpg", "umpire7_mp4-0176.jpg", "output_original_frame_00179-jpg_<uuid>.jpg"
_FRAME_NAME_PATTERN = re.compile(r"^(.+?)(?:_mp4-|_)(\d+)")


def _video_and_frame_number(frame_name):
    match = _FRAME_NAME_PATTERN.match(frame_name)

    if match is None:
        raise ValueError(f"Cannot find a frame number in {frame_name}")

    return match.group(1), int(match.group(2))


def build_clips(dataset):
    first_index_of_frame = {}
    for index, frame_name in enumerate(dataset["Frame"]):
        first_index_of_frame.setdefault(frame_name, index)

    video_to_frames = defaultdict(list)
    for frame_name, index in first_index_of_frame.items():
        video, frame_number = _video_and_frame_number(frame_name)
        video_to_frames[video].append((frame_number, index))

    clips = []

    for frames in video_to_frames.values():
        frames.sort()
        clip = [frames[0]]

        for previous, current in zip(frames, frames[1:]):
            same_clip = (
                current[0] - previous[0] <= _MAX_FRAME_GAP
                and dataset["Y"][current[1]] == dataset["Y"][previous[1]]
            )

            if not same_clip:
                clips.append(clip)
                clip = []

            clip.append(current)

        clips.append(clip)

    return [np.array([index for _, index in clip]) for clip in clips]


def build_windows(dataset, window_length):
    indices = []
    clip_ids = []

    for clip_id, clip in enumerate(build_clips(dataset)):
        if len(clip) < window_length:
            clip = np.concatenate([clip, np.repeat(clip[-1], window_length - len(clip))])

        for start in range(len(clip) - window_length + 1):
            indices.append(clip[start:start + window_length])
            clip_ids.append(clip_id)

    indices = np.array(indices)
    # Every frame in a clip has the same class and umpire, so use the last frame's 1
    last_indices = indices[:, -1]

    return {
        "Indices": indices,
        "Y": dataset["Y"][last_indices],
        "Umpire": dataset["Umpire"][last_indices],
        "Clip": np.array(clip_ids),
    }


class SignalSequenceClassifier:

    def __init__(self, window_length, num_keypoints, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=(window_length, num_keypoints, 2)),
            layers.Reshape((window_length, num_keypoints * 2)),

            layers.LSTM(64),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def fit(self, train, epochs, verbose="auto"):
        # There are too few clips to hold out a validation set for early stopping, so train for a fixed number of epochs.
        self.model.fit(
            train["X"],
            train["Y"],
            epochs=epochs,
            batch_size=32,
            verbose=verbose
        )

    def predict(self, x):
        return np.argmax(self.model.predict(x, verbose=0), axis=1)
