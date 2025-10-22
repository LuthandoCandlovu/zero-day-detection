import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer()
        self.pca = PCA(n_components=0.95)
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        self.is_fitted = False
        
    def engineer_features(self, df):
        """Advanced feature engineering with multiple techniques"""
        try:
            df_engineered = df.copy()
            
            # 1. Basic Ratio Features
            df_engineered = self._create_basic_features(df_engineered)
            
            # 2. Statistical Features
            df_engineered = self._create_statistical_features(df_engineered)
            
            # 3. Time-based Features
            if 'timestamp' in df_engineered.columns:
                df_engineered = self._create_time_features(df_engineered)
            
            # 4. Interaction Features
            df_engineered = self._create_interaction_features(df_engineered)
            
            # 5. Advanced Statistical Features
            df_engineered = self._create_advanced_statistical_features(df_engineered)
            
            return df_engineered
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return df
    
    def _create_basic_features(self, df):
        """Create basic ratio and derived features"""
        df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
        df['bytes_diff'] = abs(df['src_bytes'] - df['dst_bytes'])
        df['connection_density'] = df['count'] / (df['duration'] + 0.001)
        df['service_entropy'] = self._calculate_entropy(df['service'])
        return df
    
    def _create_statistical_features(self, df):
        """Create statistical features"""
        # Rolling statistics for time-series analysis
        for window in [5, 10, 20]:
            df[f'src_bytes_rolling_mean_{window}'] = df['src_bytes'].rolling(window=window, min_periods=1).mean()
            df[f'src_bytes_rolling_std_{window}'] = df['src_bytes'].rolling(window=window, min_periods=1).std()
            df[f'count_rolling_sum_{window}'] = df['count'].rolling(window=window, min_periods=1).sum()
        
        # Z-scores for outlier detection
        df['src_bytes_zscore'] = np.abs(stats.zscore(df['src_bytes']))
        df['dst_bytes_zscore'] = np.abs(stats.zscore(df['dst_bytes']))
        
        return df
    
    def _create_time_features(self, df):
        """Create time-based features"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features between important variables"""
        df['protocol_service'] = df['protocol_type'] * df['service']
        df['bytes_count_interaction'] = df['total_bytes'] * df['count']
        df['service_flag_interaction'] = df['service'] * df['flag']
        return df
    
    def _create_advanced_statistical_features(self, df):
        """Create advanced statistical features"""
        # Percentile-based features
        df['src_bytes_percentile'] = df['src_bytes'].rank(pct=True)
        df['dst_bytes_percentile'] = df['dst_bytes'].rank(pct=True)
        
        # Change points (simplified)
        df['src_bytes_change'] = df['src_bytes'].diff().fillna(0)
        df['count_change'] = df['count'].diff().fillna(0)
        
        return df
    
    def _calculate_entropy(self, data):
        """Calculate entropy of a distribution"""
        value_counts = data.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-9))
    
    def prepare_features(self, df, labels=None, training=True):
        """Complete feature preparation pipeline"""
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Select numeric columns for scaling
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        X = df_engineered[numeric_cols].fillna(0)
        
        if training:
            # Fit and transform
            X_scaled = self.scaler.fit_transform(X)
            X_transformed = self.power_transformer.fit_transform(X_scaled)
            
            if labels is not None:
                X_selected = self.feature_selector.fit_transform(X_transformed, labels)
            else:
                X_selected = X_transformed
                
            self.is_fitted = True
            return X_selected
        else:
            # Transform only
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with training=True first.")
            
            X_scaled = self.scaler.transform(X)
            X_transformed = self.power_transformer.transform(X_scaled)
            X_selected = self.feature_selector.transform(X_transformed)
            return X_selected
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if hasattr(self.feature_selector, 'scores_'):
            return self.feature_selector.scores_
        return None

# Backward compatibility
FeatureEngineer = AdvancedFeatureEngineer