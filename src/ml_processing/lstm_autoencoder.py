"""LSTM Autoencoder for anomaly detection and prediction.

Reconstruction-error approach: the autoencoder learns to reconstruct
normal patterns. High reconstruction error → anomaly.

Also supports sequence prediction for forecasting horizon.

Trade-offs:
+ Captures temporal dependencies and seasonality
+ Can forecast future values
+ Learns complex non-linear patterns
- Higher computational cost (training)
- Requires more data for convergence
- Less interpretable than statistical methods
- Needs GPU for efficient training on large datasets
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.core.config import get_config
from src.core.logging import get_logger
from src.ml_processing.model_manager import DetectionResult

logger = get_logger("ml.lstm_autoencoder")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warn("lstm_autoencoder.torch_unavailable")


if TORCH_AVAILABLE:

    class LSTMAutoencoder(nn.Module):
        """LSTM-based autoencoder for time series reconstruction."""

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Encoder
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Decoder
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Output projection
            self.output_proj = nn.Linear(hidden_size, input_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: encode → decode → project."""
            # Encode
            _, (hidden, cell) = self.encoder(x)

            # Repeat encoder output for decoder input
            seq_len = x.size(1)
            decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

            # Decode
            decoded, _ = self.decoder(decoder_input, (hidden, cell))

            # Project to input dimension
            return self.output_proj(decoded)

    class LSTMForecaster(nn.Module):
        """LSTM for multi-step forecasting."""

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            forecast_horizon: int = 24,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, forecast_horizon)
            self.forecast_horizon = forecast_horizon

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            # Use last timestep
            return self.fc(lstm_out[:, -1, :])


class LSTMAnomalyDetector:
    """LSTM Autoencoder anomaly detector with optional forecasting."""

    def __init__(self, config: dict | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM: pip install torch")

        cfg = (config or get_config())["ml"]["models"]["lstm_autoencoder"]
        self.model_name = "lstm_autoencoder"
        self.sequence_length = cfg.get("sequence_length", 60)
        self.hidden_size = cfg.get("hidden_size", 64)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.2)
        self.lr = cfg.get("learning_rate", 0.001)
        self.epochs = cfg.get("epochs", 50)
        self.batch_size = cfg.get("batch_size", 32)
        self.threshold_percentile = cfg.get("threshold_percentile", 95)
        self._model: LSTMAutoencoder | None = None
        self._forecaster: LSTMForecaster | None = None
        self._threshold: float = 0.0
        self._trained = False
        self._input_size: int = 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_trained(self) -> bool:
        return self._trained and self._model is not None

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Convert flat time series into overlapping sequences."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences)

    def train(self, data: np.ndarray, **kwargs) -> dict:
        """Train the LSTM autoencoder.

        Args:
            data: 2D array (n_samples, n_features) or 1D (n_samples,)

        Returns:
            Training metrics dict
        """
        start = time.time()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._input_size = data.shape[1]

        # Create sequences
        sequences = self._create_sequences(data)
        if len(sequences) < 2:
            return {"error": "insufficient data for sequences"}

        # Convert to tensors
        x_tensor = torch.FloatTensor(sequences).to(self._device)

        # Initialize model
        self._model = LSTMAutoencoder(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)

        # Train
        dataset = TensorDataset(x_tensor, x_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        losses = []

        self._model.train()
        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if _epoch % 10 == 0:
                logger.debug("lstm.train_epoch", epoch=_epoch, loss=round(avg_loss, 6))

        # Compute reconstruction error threshold on training data
        self._model.eval()
        with torch.no_grad():
            reconstructions = self._model(x_tensor)
            errors = torch.mean((x_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
        self._threshold = float(np.percentile(errors, self.threshold_percentile))

        self._trained = True
        training_time = time.time() - start

        metrics = {
            "training_time_s": round(training_time, 3),
            "final_loss": round(losses[-1], 6),
            "n_sequences": len(sequences),
            "threshold": round(self._threshold, 6),
            "device": str(self._device),
        }
        logger.info("lstm_autoencoder.trained", **metrics)
        return metrics

    def train_forecaster(self, data: np.ndarray, horizon: int = 24) -> dict:
        """Train the LSTM forecaster for prediction horizon.

        Args:
            data: 1D or 2D time series data
            horizon: Number of future steps to predict
        """
        start = time.time()
        data = np.nan_to_num(data, nan=0.0)

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._input_size = data.shape[1]

        # Create input-target pairs
        x_list, y_list = [], []
        for i in range(len(data) - self.sequence_length - horizon + 1):
            x_list.append(data[i : i + self.sequence_length])
            # Target: next `horizon` values of first feature
            y_list.append(data[i + self.sequence_length : i + self.sequence_length + horizon, 0])

        if len(x_list) < 2:
            return {"error": "insufficient data for forecasting"}

        x_tensor = torch.FloatTensor(np.array(x_list)).to(self._device)
        y_tensor = torch.FloatTensor(np.array(y_list)).to(self._device)

        self._forecaster = LSTMForecaster(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            forecast_horizon=horizon,
            dropout=self.dropout,
        ).to(self._device)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._forecaster.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._forecaster.train()
        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self._forecaster(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self._forecaster.eval()
        training_time = time.time() - start
        logger.info("lstm_forecaster.trained", time_s=round(training_time, 3), horizon=horizon)
        return {"training_time_s": round(training_time, 3), "horizon": horizon}

    def detect(self, features: np.ndarray) -> DetectionResult:
        """Detect anomaly via reconstruction error.

        Args:
            features: Sequence of shape (seq_len, features) or (features,)
        """
        if not self.is_trained:
            return DetectionResult(
                model_name=self.model_name, anomaly_score=0.0, is_anomaly=False,
                details={"error": "model not trained"},
            )

        start = time.monotonic()

        if features.ndim == 1:
            if len(features) >= self.sequence_length:
                features = features[-self.sequence_length:].reshape(1, self.sequence_length, -1 if self._input_size > 1 else 1)
            else:
                # Pad from the left
                pad = np.zeros((1, self.sequence_length - len(features), self._input_size))
                feat = features.reshape(1, -1, self._input_size)
                features = np.concatenate([pad, feat], axis=1)

        if features.ndim == 2:
            features = features.reshape(1, features.shape[0], self._input_size)

        features = np.nan_to_num(features, nan=0.0)
        x_tensor = torch.FloatTensor(features).to(self._device)

        self._model.eval()
        with torch.no_grad():
            reconstruction = self._model(x_tensor)
            error = float(torch.mean((x_tensor - reconstruction) ** 2).cpu().numpy())

        # Score: 0 = normal, 1 = anomalous
        anomaly_score = min(1.0, error / (self._threshold * 2)) if self._threshold > 0 else 0.0

        is_anomaly = error > self._threshold
        latency = (time.monotonic() - start) * 1000

        return DetectionResult(
            model_name=self.model_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            details={
                "reconstruction_error": round(error, 6),
                "threshold": round(self._threshold, 6),
            },
            latency_ms=latency,
        )

    def predict(self, features: np.ndarray, horizon: int = 24) -> np.ndarray | None:
        """Forecast future values.

        Args:
            features: Recent sequence of shape (seq_len, features)
            horizon: Number of steps to forecast

        Returns:
            Array of forecasted values, or None if forecaster not trained
        """
        if self._forecaster is None:
            return None

        if features.ndim == 1:
            features = features.reshape(1, -1, self._input_size)
        if features.ndim == 2:
            features = features.reshape(1, features.shape[0], self._input_size)

        x_tensor = torch.FloatTensor(np.nan_to_num(features, nan=0.0)).to(self._device)
        self._forecaster.eval()
        with torch.no_grad():
            forecast = self._forecaster(x_tensor).cpu().numpy()
        return forecast[0]

    def save(self, path: Path) -> None:
        if self._model is not None:
            save_dict = {
                "model_state": self._model.state_dict(),
                "threshold": self._threshold,
                "input_size": self._input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "sequence_length": self.sequence_length,
            }
            if self._forecaster:
                save_dict["forecaster_state"] = self._forecaster.state_dict()
            torch.save(save_dict, path)
            logger.info("lstm_autoencoder.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> LSTMAnomalyDetector:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        model = cls.__new__(cls)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model._input_size = checkpoint["input_size"]
        model.hidden_size = checkpoint["hidden_size"]
        model.num_layers = checkpoint["num_layers"]
        model.sequence_length = checkpoint["sequence_length"]
        model._threshold = checkpoint["threshold"]
        model.model_name = "lstm_autoencoder"
        model._device = torch.device("cpu")

        model._model = LSTMAutoencoder(
            input_size=model._input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
        )
        model._model.load_state_dict(checkpoint["model_state"])
        model._model.eval()
        model._trained = True

        if "forecaster_state" in checkpoint:
            model._forecaster = LSTMForecaster(
                input_size=model._input_size,
                hidden_size=model.hidden_size,
                num_layers=model.num_layers,
            )
            model._forecaster.load_state_dict(checkpoint["forecaster_state"])
            model._forecaster.eval()
        else:
            model._forecaster = None

        return model
