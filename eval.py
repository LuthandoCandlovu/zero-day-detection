import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import logging
from logger import setup_logger

def evaluate_model(model, features, cv_folds=5):
    """Evaluate model performance using cross-validation"""
    logger = setup_logger('evaluation')
    
    try:
        # Since we're dealing with unsupervised learning, we'll use model scores
        # In practice, you might have labeled data for evaluation
        
        # Get model scores
        predictions, scores = model.predict(features)
        
        # Calculate basic metrics
        anomaly_rate = np.mean(predictions == -1)
        avg_score = np.mean(scores)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'average_score': avg_score,
            'score_std': np.std(scores),
            'num_anomalies': np.sum(predictions == -1)
        }
        
        logger.info(f"Evaluation completed: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def generate_evaluation_report(model, features, output_path='reports/evaluation_report.json'):
    """Generate comprehensive evaluation report"""
    import json
    import os
    
    logger = setup_logger('evaluation')
    
    try:
        # Create reports directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get predictions and scores
        predictions, scores = model.predict(features)
        
        # Generate report
        report = {
            'summary': {
                'total_samples': len(features),
                'anomalies_detected': int(np.sum(predictions == -1)),
                'anomaly_rate': float(np.mean(predictions == -1))
            },
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            },
            'feature_importance': get_feature_importance(model, features)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Evaluation report saved to {output_path}")
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise

def get_feature_importance(model, features):
    """Extract feature importance if available"""
    if hasattr(model.model, 'feature_importances_'):
        return model.model.feature_importances_.tolist()
    else:
        return "Feature importance not available for this model type"