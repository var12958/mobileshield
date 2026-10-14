"""Real-Time Flow Prediction Engine.

Loads saved preprocessing pipeline and trained models from `models/`, accepts
unseen flow dictionaries or DataFrames, applies identical transformations,
and outputs classification predictions and confidence scores.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.prepare_dataset import NetworkFlowPreprocessor
except ImportError:
    from prepare_dataset import NetworkFlowPreprocessor

# Ensure joblib unpickling compatibility across modules
import __main__
setattr(__main__, "NetworkFlowPreprocessor", NetworkFlowPreprocessor)


class FlowPredictor:
    """Production inference engine for live mobile network flow classification."""

    def __init__(
        self,
        preprocessor_path: str = "models/preprocessor.joblib",
        binary_model_path: str = "models/binary/best_model.joblib",
        multiclass_model_path: str = "models/multiclass/best_model.joblib",
        label_mapping_path: str = "models/label_mapping.json"
    ):
        self.preprocessor_path = Path(preprocessor_path)
        self.binary_model_path = Path(binary_model_path)
        self.multiclass_model_path = Path(multiclass_model_path)
        self.label_mapping_path = Path(label_mapping_path)

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.preprocessor_path}")
        self.preprocessor: NetworkFlowPreprocessor = joblib.load(self.preprocessor_path)

        # Load models
        self.binary_model = joblib.load(self.binary_model_path) if self.binary_model_path.exists() else None
        self.multiclass_model = joblib.load(self.multiclass_model_path) if self.multiclass_model_path.exists() else None

        # Load label mappings
        with open(self.label_mapping_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
            self.binary_classes = {v: k for k, v in mappings["binary"].items()}
            self.multiclass_classes = {v: k for k, v in mappings["multiclass"].items()}

    def predict_flow(self, flow: Union[Dict[str, Any], pd.DataFrame], mode: str = "binary") -> Dict[str, Any]:
        """Classifies a network flow record.
        
        Args:
            flow: A dictionary of flow features or a pandas DataFrame.
            mode: "binary" or "multiclass".
            
        Returns:
            Dict containing predicted class index, label name, and class probability scores.
        """
        if isinstance(flow, dict):
            df = pd.DataFrame([flow])
        else:
            df = flow.copy()

        # Transform using fitted preprocessor
        X_scaled = self.preprocessor.transform(df)

        if mode == "binary":
            if self.binary_model is None:
                raise RuntimeError("Binary model artifact not loaded.")
            pred = self.binary_model.predict(X_scaled)[0]
            probs = self.binary_model.predict_proba(X_scaled)[0] if hasattr(self.binary_model, "predict_proba") else None
            
            label_name = self.binary_classes.get(int(pred), f"Class_{pred}")
            conf = float(probs[int(pred)]) if probs is not None else 1.0

            return {
                "mode": "binary",
                "prediction": int(pred),
                "label": label_name,
                "confidence": round(conf, 4),
                "probabilities": {self.binary_classes.get(i, f"Class_{i}"): round(float(p), 4) for i, p in enumerate(probs)} if probs is not None else {}
            }

        elif mode == "multiclass":
            if self.multiclass_model is None:
                raise RuntimeError("Multiclass model artifact not loaded.")
            pred = self.multiclass_model.predict(X_scaled)[0]
            probs = self.multiclass_model.predict_proba(X_scaled)[0] if hasattr(self.multiclass_model, "predict_proba") else None

            label_name = self.multiclass_classes.get(int(pred), f"Class_{pred}")
            conf = float(probs[int(pred)]) if probs is not None else 1.0

            return {
                "mode": "multiclass",
                "prediction": int(pred),
                "label": label_name,
                "confidence": round(conf, 4),
                "probabilities": {self.multiclass_classes.get(i, f"Class_{i}"): round(float(p), 4) for i, p in enumerate(probs)} if probs is not None else {}
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'binary' or 'multiclass'.")


def run_demo():
    """Runs a quick live verification test using a sample flow from test split."""
    print("[*] Initializing FlowPredictor...")
    predictor = FlowPredictor()

    data = np.load("data/processed/processed_splits.npz", allow_pickle=True)
    feat_names = data["feature_names"].tolist()
    sample_vec = data["X_test"][0]

    # Reconstruct dictionary
    sample_dict = {feat_names[i]: float(sample_vec[i]) for i in range(len(feat_names))}

    print("\n[*] Running Binary Inference on Test Sample:")
    bin_res = predictor.predict_flow(sample_dict, mode="binary")
    print(f"    Label:       {bin_res['label']}")
    print(f"    Confidence:  {bin_res['confidence']*100:.2f}%")
    print(f"    Probabilities: {bin_res['probabilities']}")

    print("\n[*] Running Multiclass Inference on Test Sample:")
    multi_res = predictor.predict_flow(sample_dict, mode="multiclass")
    print(f"    Label:       {multi_res['label']}")
    print(f"    Confidence:  {multi_res['confidence']*100:.2f}%")
    print(f"    Probabilities: {multi_res['probabilities']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Flow Prediction Engine")
    parser.add_argument("--demo", action="store_true", help="Run demonstration inference test")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()
