"""Neural autoencoder-style reconstruction anomaly detector."""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ReconstructionAutoencoder:
    """Compact MLP autoencoder for unordered industrial telemetry.

    The data has no timestamp/order suitable for LSTM/GRU. This model learns normal
    operating reconstructions and uses validation-normal reconstruction error as the
    threshold; it is not a sequence model.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", early_stopping=True,
                                  validation_fraction=0.15, max_iter=80, random_state=random_state,
                                  learning_rate_init=0.001, batch_size=256)
        self.threshold_: float | None = None

    def fit(self, normal_x: np.ndarray, validation_normal_x: np.ndarray) -> "ReconstructionAutoencoder":
        normal_scaled = self.scaler.fit_transform(normal_x)
        self.model.fit(normal_scaled, normal_scaled)
        validation_scaled = self.scaler.transform(validation_normal_x)
        errors = self._errors_scaled(validation_scaled)
        self.threshold_ = float(np.quantile(errors, 0.995))
        return self

    def _errors_scaled(self, x: np.ndarray) -> np.ndarray:
        rebuilt = self.model.predict(x)
        return np.mean((x - rebuilt) ** 2, axis=1)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return self._errors_scaled(self.scaler.transform(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Detector has not been fitted.")
        return (self.score_samples(x) > self.threshold_).astype(int)
