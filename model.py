import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from logger import setup_logger

class ZeroDayModel:
    """Main model class for zero-day anomaly detection"""
    
    def __init__(self, model_type='isolation_forest', **kwargs):
        self.logger = setup_logger('zero_day_model')
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model parameters
        self.model_params = {
            'isolation_forest': {
                'n_estimators': kwargs.get('n_estimators', 100),
                'contamination': kwargs.get('contamination', 0.1),
                'random_state': kwargs.get('random_state', 42)
            },
            'one_class_svm': {
                'kernel': kwargs.get('kernel', 'rbf'),
                'nu': kwargs.get('nu', 0.1),
                'gamma': kwargs.get('gamma', 'scale')
            }
        }
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'isolation_forest':
            params = self.model_params['isolation_forest']
            self.model = IsolationForest(
                n_estimators=params['n_estimators'],
                contamination=params['contamination'],
                random_state=params['random_state']
            )
        elif self.model_type == 'one_class_svm':
            params = self.model_params['one_class_svm']
            self.model = OneClassSVM(
                kernel=params['kernel'],
                nu=params['nu'],
                gamma=params['gamma']
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.logger.info(f"Initialized {self.model_type} model")
    
    def preprocess_features(self, features):
        """Preprocess features for model training/prediction"""
        if isinstance(features, pd.DataFrame):
            features = features.values
            
        # Handle missing values
        features = np.nan_to_num(features)
        
        return features
    
    def train(self, features):
        """Train the anomaly detection model"""
        try:
            # Preprocess features
            processed_features = self.preprocess_features(features)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(processed_features)
            
            # Train model
            self.model.fit(scaled_features)
            self.is_trained = True
            
            self.logger.info(f"Model trained successfully on {len(features)} samples")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def predict(self, features):
        """Predict anomalies in features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            # Preprocess features
            processed_features = self.preprocess_features(features)
            
            # Scale features
            scaled_features = self.scaler.transform(processed_features)
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(scaled_features)
            
            # Get anomaly scores
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(scaled_features)
            else:
                scores = self.model.score_samples(scaled_features)
                
            self.logger.info(f"Prediction completed: {sum(predictions == -1)} anomalies detected")
            
            return predictions, scores
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise