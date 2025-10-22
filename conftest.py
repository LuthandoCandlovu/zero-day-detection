import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_network_data():
    """Fixture for sample network data"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'source_ip': ['192.168.1.' + str(i) for i in range(100)],
        'destination_ip': ['10.0.0.' + str(i) for i in range(100)],
        'packet_size': np.random.randint(64, 1500, 100),
        'duration': np.random.uniform(0.1, 10.0, 100),
        'protocol': np.random.choice(['TCP', 'UDP', 'HTTP'], 100)
    })

@pytest.fixture
def sample_anomalous_data():
    """Fixture for sample anomalous data"""
    data = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'feature3': np.random.randn(50)
    })
    # Add some anomalies
    data.iloc[45:50] = data.iloc[45:50] * 5  # Make last 5 points anomalous
    return data