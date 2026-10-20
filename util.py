import numpy as np


def macro_f1(y_true, y_predicted, num_classes):
    """
    Mean F1 score over the classes that appear in either y_true or y_predicted (not every umpire has every class).
    Predictions of -1 (no detection) count as wrong but do not add a class.
    """
    f1_scores = []

    for c in range(num_classes):
        true_positives = np.sum((y_predicted == c) & (y_true == c))
        predicted_positives = np.sum(y_predicted == c)
        actual_positives = np.sum(y_true == c)

        if predicted_positives == 0 and actual_positives == 0:
            continue

        f1_scores.append(2 * true_positives / (predicted_positives + actual_positives))

    return np.mean(f1_scores)
