"""
ForestShield: A high-performance random forest implementation.
"""

from .ensemble.forest import RandomForestClassifier
from .feature_selection.select_from_model import SelectFromModel
from .tree.tree import DecisionTreeClassifier

__all__ = [
    "RandomForestClassifier",
    "SelectFromModel",
    "DecisionTreeClassifier",
]

__version__ = "0.1.0"
