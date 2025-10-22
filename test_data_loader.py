import pytest
import pandas as pd
import os
import sys
sys.path.append('../')
from data_loader import DataLoader

class TestDataLoader:
    def setup_method(self):
        self.loader = DataLoader()
        self.test_data_path = "tests/test_data/sample_network_data.csv"
        
    def test_load_csv_success(self):
        """Test successful CSV loading"""
        if os.path.exists(self.test_data_path):
            data = self.loader.load_csv(self.test_data_path)
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            
    def test_load_csv_file_not_found(self):
        """Test handling of missing file"""
        with pytest.raises(FileNotFoundError):
            self.loader.load_csv("nonexistent_file.csv")
            
    def test_preprocess_data(self):
        """Test data preprocessing"""
        sample_data = pd.DataFrame({
            'packet_size': [100, 200, 300],
            'duration': [1.5, 2.0, 0.5],
            'protocol': ['TCP', 'UDP', 'TCP']
        })
        
        processed = self.loader.preprocess_data(sample_data)
        assert 'protocol_encoded' in processed.columns