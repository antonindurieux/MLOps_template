import json
import random

import numpy as np


def load_dict(filepath: str) -> dict:
    """
    Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): Location of file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """
    Save a dictionary to a specific location.

    Args:
        d (dict): Data to save.
        filepath (str): Location of where to save the data.
        cls (optional): Encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): Whether to sort keys alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed: int = 42) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed (int, optional): Number to be used as the seed. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
