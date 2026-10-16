import umpire_detection, keypoint_detection, keypoint_classification, object_detection, signal_analysis_over_time

def main():
    keypoint_classifier = keypoint_classification.KeypointClassifier()
    keypoint_classifier.load_training_data_from_coco()
    pass


if __name__ == "__main__":
    main()
