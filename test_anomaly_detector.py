import pytest
import numpy as np
import sys
sys.path.append('../')
from anomaly_detector import AnomalyDetector

class TestAnomalyDetector:
    def setup_method(self):
        self.detector = AnomalyDetector()
        self.sample_features = np.random.randn(100, 5)
        
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.detector.model is None
        
    def test_train_model(self):
        """Test model training"""
        self.detector.train(self.sample_features)
        assert self.detector.model is not None
        
    def test_predict_anomalies(self):
        """Test anomaly prediction"""
        self.detector.train(self.sample_features)
        predictions = self.detector.predict(self.sample_features)
        assert len(predictions) == len(self.sample_features)
        assert set(predictions).issubset([-1, 1])