"""
Stage 2: Gaussian Process Surrogate Model.

After the DoE sweep produces (setup → lap_time) pairs, we train a GP to
predict lap time from any setup in milliseconds — no full sim needed.

Why a GP?
  - Gives uncertainty estimates (it knows where it's confident vs. guessing)
  - The uncertainty is what Bayesian Optimization uses to decide where to
    sample next: high uncertainty = worth exploring, low uncertainty + good
    prediction = exploit it.
  - Interpretable and resume-worthy for an engineering audience.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle


class LapTimeSurrogate:
    def __init__(self):
        # RBF (Radial Basis Function) kernel: smooth, continuous predictions
        # ConstantKernel scales overall magnitude
        # WhiteKernel models noise in the simulator output
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,   # try 10 random starts to find best hyperparams
            normalize_y=True,
        )
        self.scaler = StandardScaler()  # GPs are sensitive to scale; normalize inputs
        self.feature_names = None

    def fit(self, df: pd.DataFrame, target_col: str = "lap_time"):
        feature_cols = [c for c in df.columns if c != target_col]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values

        X_scaled = self.scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        print(f"GP trained on {len(y)} samples.")
        print(f"Optimized kernel: {self.gp.kernel_}")

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predict lap time(s). X shape: (n_samples, n_features)."""
        X_scaled = self.scaler.transform(X)
        return self.gp.predict(X_scaled, return_std=return_std)

    def cross_validate(self, df: pd.DataFrame, target_col: str = "lap_time", cv: int = 5):
        feature_cols = [c for c in df.columns if c != target_col]
        X = self.scaler.transform(df[feature_cols].values)
        y = df[target_col].values
        scores = cross_val_score(self.gp, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        rmse = -scores.mean()
        print(f"Cross-validation RMSE: {rmse:.4f}s ({cv}-fold)")
        return rmse

    def save(self, path: str = "surrogate.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"gp": self.gp, "scaler": self.scaler, "features": self.feature_names}, f)
        print(f"Surrogate saved to {path}")

    @classmethod
    def load(cls, path: str = "surrogate.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.gp = data["gp"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["features"]
        return obj


if __name__ == "__main__":
    df = pd.read_csv("doe_results.csv")
    surrogate = LapTimeSurrogate()
    surrogate.fit(df)
    surrogate.cross_validate(df)
    surrogate.save()
