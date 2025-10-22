import argparse
import yaml
import pandas as pd
from model import ZeroDayModel
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from logger import setup_logger
import os

def load_config(config_path='config/model_config.yaml'):
    """Load training configuration"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Zero-Day Detection Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='data/models/zero_day_model.pkl',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('train')
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # Load data
        loader = DataLoader()
        data = loader.load_csv(args.data)
        logger.info(f"Data loaded: {len(data)} records")
        
        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.create_features(data)
        logger.info(f"Features engineered: {features.shape}")
        
        # Initialize and train model
        model_config = config['model']['parameters']
        model = ZeroDayModel(
            model_type=config['model']['type'],
            **model_config
        )
        
        model.train(features)
        logger.info("Model training completed")
        
        # Save model
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        model.save_model(args.output)
        logger.info(f"Model saved to {args.output}")
        
        # Evaluate model (optional)
        if config['training'].get('cross_validation', False):
            from eval import evaluate_model
            evaluation_results = evaluate_model(model, features)
            logger.info(f"Model evaluation: {evaluation_results}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()