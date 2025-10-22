import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from logger import setup_logger

class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self, window_size=1000):
        self.logger = setup_logger('model_monitor')
        self.window_size = window_size
        self.performance_history = []
        self.drift_alerts = []
        
    def track_performance(self, y_true, y_pred, timestamp=None):
        """Track model performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
            
        accuracy = np.mean(y_true == y_pred)
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        
        performance_metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
        
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            
        self.logger.info(f"Performance tracked: {performance_metrics}")
        return performance_metrics
    
    def check_data_drift(self, current_data, reference_data):
        """Check for data drift using statistical tests"""
        drift_detected = False
        drift_metrics = {}
        
        for column in current_data.columns:
            if current_data[column].dtype in ['float64', 'int64']:
                # KS test for numerical data
                from scipy import stats
                stat, p_value = stats.ks_2samp(
                    reference_data[column].dropna(), 
                    current_data[column].dropna()
                )
                drift_metrics[column] = {
                    'ks_statistic': stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
                
                if p_value < 0.05:
                    drift_detected = True
                    self.logger.warning(f"Data drift detected in column {column}")
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now(),
                'type': 'data_drift',
                'metrics': drift_metrics
            }
            self.drift_alerts.append(alert)
            
        return drift_detected, drift_metrics
    
    def _calculate_precision(self, y_true, y_pred):
        """Calculate precision for anomaly detection"""
        tp = np.sum((y_true == -1) & (y_pred == -1))
        fp = np.sum((y_true == 1) & (y_pred == -1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _calculate_recall(self, y_true, y_pred):
        """Calculate recall for anomaly detection"""
        tp = np.sum((y_true == -1) & (y_pred == -1))
        fn = np.sum((y_true == -1) & (y_pred == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def generate_report(self):
        """Generate monitoring report"""
        report = {
            'timestamp': datetime.now(),
            'performance_metrics': self.performance_history[-1] if self.performance_history else {},
            'total_alerts': len(self.drift_alerts),
            'recent_alerts': self.drift_alerts[-5:] if self.drift_alerts else []
        }
        return report