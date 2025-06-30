import json
import os
from pathlib import Path

import joblib  # To have and load model/preprocesors of sckit-learn
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from fraud_detector import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and return a ConfigBox object.

    Args:
        path_to_yaml (str): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: For any other YAML loading error.

    Returns:
        ConfigBox: A ConfigBox object containing the YAML content.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded succesfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories if they dont't exist.

    Args:
        path_to_directories (list): List of paths to directories to create.
        verbose (bool, optional): If True, prints a message for each directory created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a JSON file. 

     Args:
        path (Path): Path to save the JSON file.
        data (dict): Data to be saved in JSON format.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: str) -> ConfigBox:
    """Loads data from a JSON file. 


    Args:
        path (str): Path to the JSON file.

    Returns:
        ConfigBox: Data as ConfigBox type.
    """
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: object, path: str):
    """
    Saves data as a binary file (e.g., trained model, preprocessor).

    Args:
        data (object): Data to be saved (e.g., model, preprocessor).
        path (str): Path to binary file.
    # """
    path = Path(path)  # Ensures that even if someone passes a str, it becomes
    # # Create directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> object:
    """Loads data from binary file.

    Args:
        path (Path): Path to binary file.

    Returns:
        object: Loaded data (e.g., model, preprocessor).
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
