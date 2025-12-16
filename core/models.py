"""
Comprehensive ML Ensemble, Calibration, and Uncertainty Quantification Module
for AI Futures Trading Bot

This module implements:
- Multiple ML model ensemble with stacking/blending
- Probability calibration (Platt, Isotonic, Temperature)
- Uncertainty quantification (Monte Carlo, Conformal, Bayesian)
- Model performance tracking and diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict
from copy import deepcopy

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, brier_score_loss, calibration_curve
)

import scipy.stats as stats
from scipy.special import softmax, expit
from scipy.optimize import minimize


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates from a model."""
    point_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    uncertainty: np.ndarray
    confidence: np.ndarray
    method: str
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for serialization."""
        return {
            'point_prediction': self.point_prediction,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'method': self.method
        }


@dataclass
class CalibrationMetrics:
    """Container for calibration performance metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float  # Brier Score
    log_loss: float
    reliability_diagram: Tuple[np.ndarray, np.ndarray]  # (mean_pred, mean_true)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier': self.brier,
            'log_loss': self.log_loss
        }


class CalibrationMethod(ABC):
    """Abstract base class for calibration methods."""
    
    @abstractmethod
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'CalibrationMethod':
        """Fit calibration method."""
        pass
    
    @abstractmethod
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        pass


class PlattCalibration(CalibrationMethod):
    """Platt scaling calibration method."""
    
    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.fitted = False
    
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'PlattCalibration':
        """Fit Platt scaling model."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_flat = y_proba[:, 1]
        else:
            y_proba_flat = y_proba.ravel()
        
        self.lr.fit(y_proba_flat.reshape(-1, 1), y_true)
        self.fitted = True
        return self
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply Platt calibration."""
        if not self.fitted:
            raise ValueError("Must fit before calibrating")
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_flat = y_proba[:, 1]
        else:
            y_proba_flat = y_proba.ravel()
        
        calibrated = self.lr.predict_proba(y_proba_flat.reshape(-1, 1))[:, 1]
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            result = np.column_stack([1 - calibrated, calibrated])
        else:
            result = calibrated
        
        return np.clip(result, 1e-7, 1 - 1e-7)


class IsotonicCalibration(CalibrationMethod):
    """Isotonic regression calibration method."""
    
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibration':
        """Fit isotonic regression model."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_flat = y_proba[:, 1]
        else:
            y_proba_flat = y_proba.ravel()
        
        self.iso.fit(y_proba_flat, y_true)
        self.fitted = True
        return self
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self.fitted:
            raise ValueError("Must fit before calibrating")
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_flat = y_proba[:, 1]
        else:
            y_proba_flat = y_proba.ravel()
        
        calibrated = self.iso.predict(y_proba_flat)
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            result = np.column_stack([1 - calibrated, calibrated])
        else:
            result = calibrated
        
        return np.clip(result, 1e-7, 1 - 1e-7)


class TemperatureScaling(CalibrationMethod):
    """Temperature scaling calibration method."""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000):
        self.temperature = 1.0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fitted = False
    
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """Fit temperature scaling."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)
        else:
            y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)
        
        def nll_loss(t):
            if t <= 0:
                return np.inf
            
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                logits = np.log(y_proba_safe)
                scaled_logits = logits / t
                probs = softmax(scaled_logits, axis=1)
                loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))
            else:
                probs = expit(np.log(y_proba_safe / (1 - y_proba_safe)) / t)
                loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            
            return loss
        
        result = minimize(nll_loss, [1.0], method='Nelder-Mead')
        self.temperature = result.x[0]
        self.fitted = True
        return self
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self.fitted:
            raise ValueError("Must fit before calibrating")
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)
            logits = np.log(y_proba_safe)
            scaled_logits = logits / self.temperature
            result = softmax(scaled_logits, axis=1)
        else:
            y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)
            logits = np.log(y_proba_safe / (1 - y_proba_safe))
            probs = expit(logits / self.temperature)
            result = probs
        
        return np.clip(result, 1e-7, 1 - 1e-7)


class CalibratedEnsembleModel:
    """Ensemble model with integrated calibration."""
    
    def __init__(
        self,
        base_models: Optional[List] = None,
        meta_model=None,
        calibration_method: str = 'temperature',
        ensemble_method: str = 'stacking',
        random_state: int = 42
    ):
        """
        Initialize calibrated ensemble model.
        
        Args:
            base_models: List of base models for ensemble
            meta_model: Meta-model for stacking (default: LogisticRegression)
            calibration_method: 'platt', 'isotonic', or 'temperature'
            ensemble_method: 'stacking' or 'blending'
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        if base_models is None:
            self.base_models = self._create_default_models()
        else:
            self.base_models = base_models
        
        self.meta_model = meta_model or LogisticRegression(random_state=random_state)
        self.calibration_method = self._get_calibration_method(calibration_method)
        self.ensemble_method = ensemble_method
        
        self.fitted = False
        self.calibrated = False
        self.scaler = StandardScaler()
        self.meta_scaler = StandardScaler()
        self.performance_metrics = defaultdict(list)
    
    @staticmethod
    def _create_default_models() -> List:
        """Create default diverse base models."""
        return [
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            AdaBoostClassifier(n_estimators=100, random_state=42),
            SVC(kernel='rbf', probability=True, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000),
            KNeighborsClassifier(n_neighbors=5)
        ]
    
    @staticmethod
    def _get_calibration_method(method: str) -> CalibrationMethod:
        """Get calibration method instance."""
        methods = {
            'platt': PlattCalibration,
            'isotonic': IsotonicCalibration,
            'temperature': TemperatureScaling
        }
        
        if method not in methods:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return methods[method]()
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> 'CalibratedEnsembleModel':
        """
        Fit ensemble and meta-model.
        
        Args:
            X: Training features
            y: Training labels
            validation_split: Fraction of data to use for meta-model training
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Fit base models
        print(f"Training {len(self.base_models)} base models...")
        for i, model in enumerate(self.base_models):
            print(f"  Training model {i+1}/{len(self.base_models)}: {type(model).__name__}")
            model.fit(X_train, y_train)
        
        # Generate meta features
        print("Generating meta-features...")
        meta_features_train = self._generate_meta_features(X_train)
        meta_features_val = self._generate_meta_features(X_val)
        
        # Scale meta features
        meta_features_train_scaled = self.meta_scaler.fit_transform(meta_features_train)
        meta_features_val_scaled = self.meta_scaler.transform(meta_features_val)
        
        # Fit meta-model
        print("Fitting meta-model...")
        self.meta_model.fit(meta_features_train_scaled, y_train)
        
        # Fit calibration
        print("Fitting calibration...")
        meta_pred_proba = self.meta_model.predict_proba(meta_features_val_scaled)
        self.calibration_method.fit(meta_pred_proba, y_val)
        
        self.fitted = True
        self.calibrated = True
        
        # Compute metrics on validation set
        self._update_metrics('validation', X_val, y_val)
        
        return self
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base models."""
        meta_features = []
        
        for model in self.base_models:
            try:
                proba = model.predict_proba(X)
                if proba.ndim > 1:
                    meta_features.append(proba)
                else:
                    meta_features.append(proba.reshape(-1, 1))
            except AttributeError:
                # For models without predict_proba, use decision_function
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(X)
                    if decision.ndim == 1:
                        decision = decision.reshape(-1, 1)
                    meta_features.append(expit(decision))
                else:
                    # Fallback to predictions
                    pred = model.predict(X).reshape(-1, 1)
                    meta_features.append(pred.astype(float))
        
        return np.hstack(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with calibration."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        meta_features = self._generate_meta_features(X)
        meta_features_scaled = self.meta_scaler.transform(meta_features)
        proba = self.meta_model.predict_proba(meta_features_scaled)
        
        if self.calibrated:
            proba = self.calibration_method.calibrate(proba)
        
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        
        if proba.ndim > 1 and proba.shape[1] > 1:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return (proba >= threshold).astype(int)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        uncertainty_method: str = 'ensemble_variance',
        confidence_level: float = 0.95
    ) -> UncertaintyEstimate:
        """
        Predict with uncertainty quantification.
        
        Args:
            X: Input features
            uncertainty_method: 'ensemble_variance', 'monte_carlo', or 'conformal'
            confidence_level: Confidence level for intervals (0-1)
        
        Returns:
            UncertaintyEstimate object with predictions and bounds
        """
        if uncertainty_method == 'ensemble_variance':
            return self._uncertainty_ensemble_variance(X, confidence_level)
        elif uncertainty_method == 'monte_carlo':
            return self._uncertainty_monte_carlo(X, confidence_level)
        elif uncertainty_method == 'conformal':
            return self._uncertainty_conformal(X, confidence_level)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
    
    def _uncertainty_ensemble_variance(
        self,
        X: np.ndarray,
        confidence_level: float
    ) -> UncertaintyEstimate:
        """Estimate uncertainty from ensemble variance."""
        # Get predictions from all base models
        predictions = []
        
        for model in self.base_models:
            try:
                proba = model.predict_proba(X)
                if proba.ndim > 1:
                    predictions.append(proba[:, 1])
                else:
                    predictions.append(proba)
            except AttributeError:
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(X)
                    if decision.ndim > 1:
                        decision = decision[:, 1]
                    predictions.append(expit(decision))
                else:
                    predictions.append(model.predict(X).astype(float))
        
        predictions = np.array(predictions)
        
        # Point prediction (mean)
        point_pred = np.mean(predictions, axis=0)
        
        # Uncertainty (std)
        uncertainty = np.std(predictions, axis=0)
        
        # Confidence (inverse of uncertainty)
        confidence = 1 / (1 + uncertainty)
        
        # Confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = np.clip(point_pred - z_score * uncertainty, 0, 1)
        upper_bound = np.clip(point_pred + z_score * uncertainty, 0, 1)
        
        return UncertaintyEstimate(
            point_prediction=point_pred,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty=uncertainty,
            confidence=confidence,
            method='ensemble_variance'
        )
    
    def _uncertainty_monte_carlo(
        self,
        X: np.ndarray,
        confidence_level: float,
        n_iterations: int = 100
    ) -> UncertaintyEstimate:
        """Estimate uncertainty using Monte Carlo dropout."""
        predictions = []
        
        for _ in range(n_iterations):
            # Random subsampling of base models
            n_models = max(1, len(self.base_models) // 2)
            sampled_models = np.random.choice(
                self.base_models,
                size=n_models,
                replace=False
            )
            
            iter_preds = []
            for model in sampled_models:
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim > 1:
                        iter_preds.append(proba[:, 1])
                    else:
                        iter_preds.append(proba)
                except AttributeError:
                    if hasattr(model, 'decision_function'):
                        decision = model.decision_function(X)
                        if decision.ndim > 1:
                            decision = decision[:, 1]
                        iter_preds.append(expit(decision))
                    else:
                        iter_preds.append(model.predict(X).astype(float))
            
            predictions.append(np.mean(iter_preds, axis=0))
        
        predictions = np.array(predictions)
        
        # Point prediction (mean)
        point_pred = np.mean(predictions, axis=0)
        
        # Uncertainty (std across iterations)
        uncertainty = np.std(predictions, axis=0)
        
        # Confidence
        confidence = 1 / (1 + uncertainty)
        
        # Percentile-based confidence intervals
        alpha = (1 - confidence_level) / 2
        lower_bound = np.percentile(predictions, alpha * 100, axis=0)
        upper_bound = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return UncertaintyEstimate(
            point_prediction=point_pred,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty=uncertainty,
            confidence=confidence,
            method='monte_carlo'
        )
    
    def _uncertainty_conformal(
        self,
        X: np.ndarray,
        confidence_level: float
    ) -> UncertaintyEstimate:
        """Estimate uncertainty using conformal prediction."""
        # Point prediction
        point_pred = self.predict_proba(X)
        if point_pred.ndim > 1:
            point_pred = point_pred[:, 1]
        
        # For conformal intervals, use ensemble variance as nonconformity measure
        uncertainty_estimate = self._uncertainty_ensemble_variance(X, confidence_level)
        
        # Adapt bounds based on confidence level
        margin = uncertainty_estimate.uncertainty * stats.norm.ppf((1 + confidence_level) / 2)
        
        return UncertaintyEstimate(
            point_prediction=point_pred,
            lower_bound=np.clip(point_pred - margin, 0, 1),
            upper_bound=np.clip(point_pred + margin, 0, 1),
            uncertainty=uncertainty_estimate.uncertainty,
            confidence=uncertainty_estimate.confidence,
            method='conformal'
        )
    
    def evaluate_calibration(self, X: np.ndarray, y: np.ndarray) -> CalibrationMetrics:
        """Evaluate calibration quality."""
        y_proba = self.predict_proba(X)
        
        if y_proba.ndim > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        # Expected Calibration Error
        ece = self._calculate_ece(y_proba_pos, y)
        
        # Maximum Calibration Error
        mce = self._calculate_mce(y_proba_pos, y)
        
        # Brier Score
        brier = brier_score_loss(y, y_proba_pos)
        
        # Log Loss
        log_loss_val = log_loss(y, y_proba_pos)
        
        # Reliability diagram
        prob_true, prob_pred = calibration_curve(
            y, y_proba_pos, n_bins=10, strategy='uniform'
        )
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier=brier,
            log_loss=log_loss_val,
            reliability_diagram=(prob_pred, prob_true)
        )
    
    @staticmethod
    def _calculate_ece(y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if np.any(in_bin):
                prob_in_bin = y_proba[in_bin]
                accuracy_in_bin = y_true[in_bin]
                
                avg_confidence = np.mean(prob_in_bin)
                accuracy = np.mean(accuracy_in_bin)
                
                ece += np.abs(avg_confidence - accuracy) * np.mean(in_bin)
        
        return ece
    
    @staticmethod
    def _calculate_mce(y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if np.any(in_bin):
                prob_in_bin = y_proba[in_bin]
                accuracy_in_bin = y_true[in_bin]
                
                avg_confidence = np.mean(prob_in_bin)
                accuracy = np.mean(accuracy_in_bin)
                
                mce = max(mce, np.abs(avg_confidence - accuracy))
        
        return mce
    
    def _update_metrics(self, split: str, X: np.ndarray, y: np.ndarray) -> None:
        """Update performance metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        if y_proba.ndim > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, y_proba_pos),
            'brier': brier_score_loss(y, y_proba_pos),
            'log_loss': log_loss(y, y_proba_pos)
        }
        
        for metric_name, value in metrics.items():
            self.performance_metrics[f'{split}_{metric_name}'].append(value)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get performance metrics as DataFrame."""
        return pd.DataFrame(self.performance_metrics)
    
    def get_base_model_weights(self) -> Dict[str, float]:
        """Get approximate weights of base models in ensemble."""
        try:
            weights = self.meta_model.coef_[0]
            model_names = [type(m).__name__ for m in self.base_models]
            
            # Normalize weights
            weights_norm = weights / np.sum(np.abs(weights))
            
            return dict(zip(model_names, weights_norm.tolist()))
        except (AttributeError, IndexError):
            return {}


class BayesianEnsembleModel:
    """Bayesian ensemble with uncertainty quantification."""
    
    def __init__(self, base_models: Optional[List] = None, n_iterations: int = 100):
        """
        Initialize Bayesian ensemble.
        
        Args:
            base_models: List of base models
            n_iterations: Number of MCMC iterations
        """
        self.base_models = base_models or self._create_default_models()
        self.n_iterations = n_iterations
        self.posterior_samples = []
        self.fitted = False
    
    @staticmethod
    def _create_default_models() -> List:
        """Create default diverse base models."""
        return [
            RandomForestClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42),
            LogisticRegression(random_state=42)
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianEnsembleModel':
        """Fit Bayesian ensemble."""
        print("Fitting Bayesian ensemble...")
        
        for model in self.base_models:
            model.fit(X, y)
        
        # Generate posterior samples through bootstrap
        for i in range(self.n_iterations):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            predictions = []
            for model in self.base_models:
                try:
                    proba = model.predict_proba(X_boot)
                    predictions.append(proba[:, 1] if proba.ndim > 1 else proba)
                except:
                    predictions.append(model.predict(X_boot).astype(float))
            
            self.posterior_samples.append(np.mean(predictions, axis=0))
        
        self.fitted = True
        return self
    
    def predict_with_posterior(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions with posterior distribution."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = []
        for model in self.base_models:
            try:
                proba = model.predict_proba(X)
                predictions.append(proba[:, 1] if proba.ndim > 1 else proba)
            except:
                predictions.append(model.predict(X).astype(float))
        
        point_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            'mean': point_pred,
            'std': std_pred,
            'lower_95': np.clip(point_pred - 1.96 * std_pred, 0, 1),
            'upper_95': np.clip(point_pred + 1.96 * std_pred, 0, 1)
        }


class StackingEnsembleRegressor:
    """Stacking ensemble for regression with uncertainty."""
    
    def __init__(
        self,
        base_models: Optional[List] = None,
        meta_model=None,
        random_state: int = 42
    ):
        """Initialize stacking regressor."""
        self.base_models = base_models or self._create_default_models()
        self.meta_model = meta_model or Ridge()
        self.random_state = random_state
        self.fitted = False
        self.meta_scaler = StandardScaler()
    
    @staticmethod
    def _create_default_models() -> List:
        """Create default base regressors."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        
        return [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            SVR(kernel='rbf'),
            LinearRegression()
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Fit stacking regressor."""
        n_samples = len(X)
        split_idx = int(n_samples * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Fit base models
        for model in self.base_models:
            model.fit(X_train, y_train)
        
        # Generate meta-features
        meta_features_train = np.column_stack([
            model.predict(X_train) for model in self.base_models
        ])
        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.base_models
        ])
        
        # Scale and fit meta-model
        meta_features_train_scaled = self.meta_scaler.fit_transform(meta_features_train)
        meta_features_val_scaled = self.meta_scaler.transform(meta_features_val)
        
        self.meta_model.fit(meta_features_train_scaled, y_val)
        self.fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking regressor."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        meta_features_scaled = self.meta_scaler.transform(meta_features)
        
        return self.meta_model.predict(meta_features_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate:
        """Predict with uncertainty."""
        predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        point_pred = np.mean(predictions, axis=1)
        uncertainty = np.std(predictions, axis=1)
        confidence = 1 / (1 + uncertainty)
        
        z_score = 1.96
        lower_bound = point_pred - z_score * uncertainty
        upper_bound = point_pred + z_score * uncertainty
        
        return UncertaintyEstimate(
            point_prediction=point_pred,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty=uncertainty,
            confidence=confidence,
            method='stacking'
        )


class AdaptiveWeightingEnsemble:
    """Ensemble with adaptive weighting based on recent performance."""
    
    def __init__(self, base_models: Optional[List] = None, window_size: int = 20):
        """
        Initialize adaptive ensemble.
        
        Args:
            base_models: List of base models
            window_size: Size of performance window for weight adaptation
        """
        self.base_models = base_models or self._create_default_models()
        self.window_size = window_size
        self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        self.performance_history = defaultdict(list)
        self.fitted = False
    
    @staticmethod
    def _create_default_models() -> List:
        """Create default diverse base models."""
        return [
            RandomForestClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42),
            LogisticRegression(random_state=42)
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all base models."""
        for model in self.base_models:
            model.fit(X, y)
        
        self.fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict with adaptive weights."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = []
        for model in self.base_models:
            try:
                proba = model.predict_proba(X)
                if proba.ndim > 1:
                    predictions.append(proba[:, 1])
                else:
                    predictions.append(proba)
            except AttributeError:
                predictions.append(model.predict(X).astype(float))
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return np.column_stack([1 - weighted_pred, weighted_pred])
    
    def update_weights(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Update model weights based on recent performance."""
        for i, model in enumerate(self.base_models):
            try:
                proba = model.predict_proba(np.array([]))  # Dummy call
                accuracy = accuracy_score(y_true, y_pred)
            except:
                accuracy = 0
            
            self.performance_history[i].append(accuracy)
            
            # Keep only recent history
            if len(self.performance_history[i]) > self.window_size:
                self.performance_history[i].pop(0)
        
        # Update weights based on recent performance
        recent_scores = [
            np.mean(self.performance_history[i][-self.window_size:])
            for i in range(len(self.base_models))
        ]
        
        # Normalize to get weights
        recent_scores = np.array(recent_scores)
        self.weights = (recent_scores + 1e-6) / np.sum(recent_scores + 1e-6)


# Utility functions
def compare_calibration_methods(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    test_split: float = 0.3
) -> Dict[str, CalibrationMetrics]:
    """Compare different calibration methods."""
    n_samples = len(y_true)
    split_idx = int(n_samples * (1 - test_split))
    
    y_proba_train = y_proba[:split_idx]
    y_true_train = y_true[:split_idx]
    y_proba_test = y_proba[split_idx:]
    y_true_test = y_true[split_idx:]
    
    methods = {
        'platt': PlattCalibration(),
        'isotonic': IsotonicCalibration(),
        'temperature': TemperatureScaling()
    }
    
    results = {}
    
    for name, method in methods.items():
        method.fit(y_proba_train, y_true_train)
        y_proba_cal = method.calibrate(y_proba_test)
        
        if y_proba_cal.ndim > 1:
            y_proba_pos = y_proba_cal[:, 1]
        else:
            y_proba_pos = y_proba_cal
        
        ece = CalibratedEnsembleModel._calculate_ece(y_proba_pos, y_true_test)
        mce = CalibratedEnsembleModel._calculate_mce(y_proba_pos, y_true_test)
        brier = brier_score_loss(y_true_test, y_proba_pos)
        log_loss_val = log_loss(y_true_test, y_proba_pos)
        prob_true, prob_pred = calibration_curve(y_true_test, y_proba_pos, n_bins=10)
        
        results[name] = CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier=brier,
            log_loss=log_loss_val,
            reliability_diagram=(prob_pred, prob_true)
        )
    
    return results


def ensemble_feature_importance(
    ensemble_model: CalibratedEnsembleModel,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Extract feature importance from ensemble base models."""
    importances = []
    
    for model in ensemble_model.base_models:
        if hasattr(model, 'feature_importances_'):
            importances.append(model.feature_importances_)
    
    if not importances:
        return pd.DataFrame()
    
    importances = np.mean(importances, axis=0)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df
