import pandas as pd
import numpy as np
import requests
from io import StringIO
import zipfile
import os
from datetime import datetime, timedelta

class AdvancedDataLoader:
    def __init__(self):
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        
        self.attack_types = {
            'normal': 0,
            'dos': 1,
            'probe': 2,
            'r2l': 3,
            'u2r': 4
        }
    
    def generate_advanced_sample_data(self, n_samples=5000):
        """Generate sophisticated synthetic network data with realistic patterns"""
        np.random.seed(42)
        
        # Base distributions for normal traffic
        data = {
            'duration': np.random.exponential(0.5, n_samples),
            'protocol_type': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            'service': np.random.choice(range(70), n_samples, p=self._get_service_distribution()),
            'flag': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'src_bytes': self._generate_bytes(n_samples, 'src'),
            'dst_bytes': self._generate_bytes(n_samples, 'dst'),
            'count': np.random.poisson(8, n_samples),
            'srv_count': np.random.poisson(6, n_samples),
            'dst_host_count': np.random.poisson(12, n_samples),
            'dst_host_srv_count': np.random.poisson(10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Inject sophisticated anomalies
        df, labels = self._inject_advanced_anomalies(df, n_samples)
        
        # Add timestamp for time series analysis
        df['timestamp'] = self._generate_timestamps(n_samples)
        
        return df, labels
    
    def _get_service_distribution(self):
        """Realistic service distribution"""
        dist = np.ones(70) * 0.5  # Base probability
        # Common services get higher probability
        dist[0:10] = 3.0  # HTTP, HTTPS, SSH, etc.
        dist[20:25] = 2.0  # DNS, SMTP, etc.
        return dist / dist.sum()
    
    def _generate_bytes(self, n_samples, direction):
        """Generate realistic byte distributions"""
        if direction == 'src':
            # Most connections have small src bytes (requests)
            base = np.random.lognormal(6, 2, n_samples)
        else:
            # dst bytes can be larger (responses)
            base = np.random.lognormal(7, 2.5, n_samples)
        return base
    
    def _inject_advanced_anomalies(self, df, n_samples):
        """Inject sophisticated anomaly patterns"""
        labels = np.zeros(n_samples)
        anomaly_indices = []
        
        # 1. DDoS Attack Pattern
        ddos_count = int(n_samples * 0.03)
        ddos_indices = np.random.choice(n_samples, ddos_count, replace=False)
        df.loc[ddos_indices, 'src_bytes'] *= 0.1  # Small packets
        df.loc[ddos_indices, 'dst_bytes'] *= 0.1
        df.loc[ddos_indices, 'count'] *= 20       # High connection rate
        df.loc[ddos_indices, 'duration'] *= 0.1   # Short duration
        anomaly_indices.extend(ddos_indices)
        
        # 2. Port Scanning
        scan_count = int(n_samples * 0.02)
        scan_indices = np.random.choice(
            [i for i in range(n_samples) if i not in ddos_indices], 
            scan_count, replace=False
        )
        df.loc[scan_indices, 'service'] = np.random.randint(60, 70, scan_count)  # Uncommon ports
        df.loc[scan_indices, 'flag'] = 0  # SYN flags
        df.loc[scan_indices, 'duration'] *= 0.2
        anomaly_indices.extend(scan_indices)
        
        # 3. Data Exfiltration
        exfil_count = int(n_samples * 0.015)
        exfil_indices = np.random.choice(
            [i for i in range(n_samples) if i not in ddos_indices + scan_indices],
            exfil_count, replace=False
        )
        df.loc[exfil_indices, 'src_bytes'] *= 1000  # Large uploads
        df.loc[exfil_indices, 'dst_bytes'] *= 0.1
        df.loc[exfil_indices, 'duration'] *= 5      # Long connections
        anomaly_indices.extend(exfil_indices)
        
        # 4. Brute Force Attacks
        brute_count = int(n_samples * 0.01)
        brute_indices = np.random.choice(
            [i for i in range(n_samples) if i not in anomaly_indices],
            brute_count, replace=False
        )
        df.loc[brute_indices, 'count'] *= 15
        df.loc[brute_indices, 'srv_count'] *= 12
        df.loc[brute_indices, 'duration'] *= 0.3
        anomaly_indices.extend(brute_indices)
        
        labels[anomaly_indices] = 1
        return df, labels
    
    def _generate_timestamps(self, n_samples):
        """Generate realistic timestamps"""
        start_date = datetime.now() - timedelta(days=7)
        return [start_date + timedelta(hours=i*0.1) for i in range(n_samples)]
    
    def load_realistic_dataset(self):
        """Main method to load data"""
        return self.generate_advanced_sample_data()

# Backward compatibility
NetworkDataLoader = AdvancedDataLoader