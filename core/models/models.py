"""
models.py — Model registry for the PANNs pipeline.

Provides get_model_class() to instantiate the Cnn14 model.
"""

from panns import Cnn14


def get_model_class(model_name):
    """
    Get model class based on model name.

    Args:
        model_name: Name of the model architecture ('panns').

    Returns:
        The model class to be instantiated.
    """
    model_lower = model_name.lower()

    if model_lower == 'panns':
        return Cnn14
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: ['panns']"
        )
