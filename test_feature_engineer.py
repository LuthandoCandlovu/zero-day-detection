import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from feature_engineer import FeatureEngineer

class TestFeatureEngineer:
    def setup_method(self):
        self.engineer = FeatureEngineer()
        self.sample_data = pd.DataFrame({
            'packet_size': [100, 200, 300, 400, 500],
            'duration': [1.0, 2.0, 3.0, 4.0, 5.0],
            'protocol': ['TCP', 'UDP', 'TCP', 'HTTP', 'HTTPS']
        })
        
    def test_create_statistical_features(self):
        """Test statistical feature creation"""
        features = self.engineer.create_statistical_features(
            self.sample_data[['packet_size', 'duration']]
        )
        expected_features = ['mean', 'std', 'min', 'max', 'median']
        for feat in expected_features:
            assert any(feat in col for col in features.columns)
            
    def test_handle_categorical_features(self):
        """Test categorical feature encoding"""
        encoded_data = self.engineer.handle_categorical_features(
            self.sample_data, ['protocol']
        )
        assert 'protocol_TCP' in encoded_data.columns