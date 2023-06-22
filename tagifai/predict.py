import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: float, index: int) -> np.ndarray:
    """
    Custom predict function that defaults
    to an index if conditions are not met.

    Args:
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Minimum softmax score to predict majority class.
        index (int): Label index to use if custom conditions is not met.

    Returns:
        np.ndarray: Predicted label indices.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(texts: list[str], artifacts: dict) -> list[str]:
    """
    Predict tags for given texts.

    Args:
        texts (list[str]): Raw input texts to classify.
        artifacts (dict): Artifacts from a run.

    Returns:
        list[str]: Predictions for input texts.
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tag": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
