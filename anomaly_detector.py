import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(
                n_estimators=200,
                contamination=0.1,
                random_state=42,
                max_features=1.0,
                bootstrap=False,
                verbose=0
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1,
                cache_size=1000,
                verbose=False
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=35,
                contamination=0.1,
                novelty=True,
                metric='euclidean'
            )
        }
        self.ensemble_weights = None
        self.best_model = None
        self.best_score = -1
        self.training_history = []
        
    def train_models(self, X_train, y_train=None):
        """Train multiple anomaly detection models with advanced techniques"""
        print("🚀 Training advanced anomaly detection models...")
        
        trained_models = {}
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                print(f"📊 Training {name.replace('_', ' ').title()}...")
                
                if name == 'local_outlier_factor':
                    model.fit(X_train)
                    # For LOF, we need to use negative outlier factor scores
                    scores = -model.negative_outlier_factor_
                else:
                    model.fit(X_train)
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_train)
                    else:
                        scores = model.predict(X_train)
                
                trained_models[name] = model
                model_scores[name] = scores
                
                print(f"✅ {name.replace('_', ' ').title()} trained successfully")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        # Calculate ensemble weights based on cross-validation
        if y_train is not None:
            self.ensemble_weights = self._calculate_ensemble_weights(trained_models, X_train, y_train)
        
        return trained_models
    
    def _calculate_ensemble_weights(self, models, X, y):
        """Calculate weights for model ensemble using cross-validation"""
        print("🤝 Calculating ensemble weights...")
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        weights = {}
        
        for name, model in models.items():
            try:
                # Use cross-validation to get performance estimate
                scores = cross_val_score(model, X, y, cv=kfold, scoring='f1')
                weights[name] = np.mean(scores)
                print(f"   {name}: F1 = {weights[name]:.3f}")
            except:
                weights[name] = 0.5  # Default weight
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation with multiple metrics"""
        print("\n🎯 Evaluating models comprehensively...")
        
        best_model_name = None
        best_f1 = -1
        evaluation_results = {}
        
        for name, model in models.items():
            try:
                # Get predictions
                if name == 'local_outlier_factor':
                    predictions = model.predict(X_test)
                    anomaly_scores = -model.negative_outlier_factor_
                else:
                    predictions = model.predict(X_test)
                    anomaly_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else predictions
                
                # Convert predictions
                predictions_binary = (predictions == -1).astype(int)
                
                # Calculate comprehensive metrics
                precision = precision_score(y_test, predictions_binary, zero_division=0)
                recall = recall_score(y_test, predictions_binary, zero_division=0)
                f1 = f1_score(y_test, predictions_binary, zero_division=0)
                
                # ROC AUC if possible
                try:
                    roc_auc = roc_auc_score(y_test, anomaly_scores)
                except:
                    roc_auc = 0.0
                
                # Store results
                evaluation_results[name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'predictions': predictions_binary,
                    'scores': anomaly_scores
                }
                
                print(f"\n📊 {name.replace('_', ' ').title()} Results:")
                print(f"   ✅ Precision: {precision:.3f}")
                print(f"   ✅ Recall:    {recall:.3f}")
                print(f"   ✅ F1-Score:  {f1:.3f}")
                print(f"   ✅ ROC-AUC:   {roc_auc:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
                    self.best_model = model
                    self.best_score = f1
                    
            except Exception as e:
                print(f"❌ Error evaluating {name}: {e}")
        
        # Ensemble prediction
        if self.ensemble_weights:
            ensemble_predictions = self._ensemble_predict(models, evaluation_results, X_test)
            ensemble_f1 = f1_score(y_test, ensemble_predictions, zero_division=0)
            print(f"\n🏆 Ensemble F1-Score: {ensemble_f1:.3f}")
            
            if ensemble_f1 > best_f1:
                best_f1 = ensemble_f1
                best_model_name = 'ensemble'
                self.best_model = models
                self.best_score = ensemble_f1
        
        print(f"\n🎖️  Best Model: {best_model_name} with F1-Score: {best_f1:.3f}")
        
        # Store training history
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'best_model': best_model_name,
            'best_score': best_f1,
            'evaluation_results': evaluation_results
        })
        
        return best_model_name, evaluation_results
    
    def _ensemble_predict(self, models, evaluation_results, X):
        """Create ensemble predictions"""
        ensemble_score = np.zeros(len(X))
        
        for name, model in models.items():
            if name in evaluation_results:
                weight = self.ensemble_weights.get(name, 0.1)
                
                if name == 'local_outlier_factor':
                    scores = -model.negative_outlier_factor_
                else:
                    scores = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict(X)
                
                # Normalize scores and apply weight
                if len(scores) > 0:
                    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                    ensemble_score += weight * scores_normalized
        
        # Convert to binary predictions
        threshold = 0.5
        return (ensemble_score > threshold).astype(int)
    
    def predict_anomalies(self, X, method='best'):
        """Advanced prediction with multiple options"""
        if method == 'best' and self.best_model is not None:
            model = self.best_model
        elif method in self.models:
            model = self.models[method]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            if method == 'ensemble' or (isinstance(model, dict) and method == 'best'):
                # Ensemble prediction
                predictions = self._ensemble_predict(model, {}, X)
                scores = np.zeros(len(X))  # Placeholder for ensemble scores
            else:
                # Single model prediction
                if method == 'local_outlier_factor' or (hasattr(model, '_estimator_type') and model._estimator_type == 'outlier_detector'):
                    predictions = model.predict(X)
                    scores = -model.negative_outlier_factor_ if hasattr(model, 'negative_outlier_factor_') else model.decision_function(X)
                else:
                    predictions = model.predict(X)
                    scores = model.decision_function(X) if hasattr(model, 'decision_function') else predictions
            
            # Calculate confidence scores
            confidence = self._calculate_confidence(scores, predictions)
            
            return predictions, scores, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _calculate_confidence(self, scores, predictions):
        """Calculate confidence scores for predictions"""
        if len(scores) == 0:
            return np.array([])
        
        # Normalize scores to [0, 1] range
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
        # For anomaly predictions, higher absolute score means higher confidence
        confidence = np.abs(scores_normalized - 0.5) * 2  # Map to [0, 1]
        
        return confidence
    
    def get_detailed_analysis(self, X, feature_names=None):
        """Get detailed analysis of predictions"""
        predictions, scores, confidence = self.predict_anomalies(X)
        
        analysis = {
            'predictions': predictions,
            'anomaly_scores': scores,
            'confidence': confidence,
            'risk_level': self._assess_risk_level(scores),
            'anomaly_count': np.sum(predictions == -1) if len(predictions) > 0 else 0,
            'anomaly_percentage': (np.sum(predictions == -1) / len(predictions) * 100) if len(predictions) > 0 else 0
        }
        
        return analysis
    
    def _assess_risk_level(self, scores):
        """Assess risk level based on anomaly scores"""
        if len(scores) == 0:
            return "Unknown"
        
        max_score = np.max(np.abs(scores))
        
        if max_score > 2.0:
            return "Critical"
        elif max_score > 1.0:
            return "High"
        elif max_score > 0.5:
            return "Medium"
        else:
            return "Low"
    
    def save_model(self, filepath):
        """Save the trained model with metadata"""
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'ensemble_weights': self.ensemble_weights,
                'best_score': self.best_score,
                'training_history': self.training_history,
                'timestamp': pd.Timestamp.now()
            }
            joblib.dump(model_data, filepath)
            print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model with metadata"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.ensemble_weights = model_data.get('ensemble_weights')
        self.best_score = model_data.get('best_score', -1)
        self.training_history = model_data.get('training_history', [])
        print(f"📂 Model loaded from {filepath} (Score: {self.best_score:.3f})")

# Backward compatibility
AnomalyDetector = AdvancedAnomalyDetector