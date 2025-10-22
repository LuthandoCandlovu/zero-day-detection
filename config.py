# Configuration settings for the application
import os
from datetime import datetime

class Config:
    # App Settings
    APP_NAME = "AI Zero-Day Attack Detection"
    VERSION = "2.0.0"
    DEBUG = True
    
    # Model Settings
    MODEL_CONFIGS = {
        'isolation_forest': {
            'n_estimators': 200,
            'contamination': 0.1,
            'random_state': 42,
            'max_features': 1.0,
            'bootstrap': False
        },
        'one_class_svm': {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.1,
            'cache_size': 1000
        },
        'local_outlier_factor': {
            'n_neighbors': 35,
            'contamination': 0.1,
            'novelty': True,
            'metric': 'euclidean'
        }
    }
    
    # Feature Engineering
    FEATURE_GROUPS = {
        'basic': ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes'],
        'advanced': ['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count'],
        'statistical': ['bytes_ratio', 'connection_density', 'total_bytes', 'byte_entropy']
    }
    
    # Visualization
    COLOR_SCHEME = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffbb78',
        'info': '#17becf'
    }
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.5
    }

config = Config()