from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantModel:
    """A trivial model that always predicts a constant value."""

    y_value: float = 1.5

    def predict_one(self, height_cm: float) -> float:
        # Intentionally ignore the input.
        _ = height_cm
        return float(self.y_value)


@dataclass(frozen=True)
class MeanModel:
    """
    Model that always predicts the mean height (in meters) computed from the dataset.
    """

    mean_value: float

    def predict_one(self, height_cm: float) -> float:
        # Intentionally ignore the input.
        _ = height_cm
        return float(self.mean_value)

