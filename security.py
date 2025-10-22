import hashlib
import hmac
import os
from datetime import datetime, timedelta
import base64
import logging
from logger import setup_logger

class SecurityManager:
    """Manage security aspects of the zero-day detection system"""
    
    def __init__(self, secret_key=None):
        self.logger = setup_logger('security')
        self.secret_key = secret_key or os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
        self.allowed_ips = self._load_allowed_ips()
        
    def _load_allowed_ips(self):
        """Load allowed IP addresses from configuration"""
        return ['127.0.0.1', '192.168.1.0/24', '10.0.0.0/8']
    
    def validate_input_data(self, data):
        """Validate and sanitize input data"""
        try:
            # Check for SQL injection patterns
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION']
            data_str = str(data).upper()
            
            for keyword in sql_keywords:
                if keyword in data_str:
                    self.logger.warning(f"Potential SQL injection detected: {keyword}")
                    return False
            
            # Check data size limits
            if len(str(data)) > 1000000:  # 1MB limit
                self.logger.warning("Input data too large")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            return False
    
    def generate_audit_log(self, event_type, user_id, details):
        """Generate security audit log entry"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': event_type,
            'user_id': user_id,
            'details': details
        }
        
        self.logger.info(f"Audit event: {event_type}", extra={'audit_data': audit_entry})
        return audit_entry
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data using base64 (simple example)"""
        try:
            if isinstance(data, str):
                data = data.encode()
            encoded = base64.b64encode(data).decode()
            return f"encrypted:{encoded}"
        except Exception as e:
            self.logger.error(f"Data encryption failed: {str(e)}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data"""
        try:
            if encrypted_data.startswith('encrypted:'):
                encoded = encrypted_data[10:]  # Remove 'encrypted:' prefix
                decoded = base64.b64decode(encoded).decode()
                return decoded
            return encrypted_data
        except Exception as e:
            self.logger.error(f"Data decryption failed: {str(e)}")
            return encrypted_data
    
    def check_access_control(self, user_role, required_role):
        """Check if user has required role for access"""
        role_hierarchy = {
            'admin': 3,
            'analyst': 2,
            'viewer': 1
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def generate_api_token(self, user_id, expires_hours=24):
        """Generate simple API token (replace with JWT in production)"""
        token_data = f"{user_id}:{datetime.utcnow().isoformat()}:{expires_hours}"
        token = base64.b64encode(token_data.encode()).decode()
        return token
    
    def verify_api_token(self, token):
        """Verify API token validity"""
        try:
            decoded = base64.b64decode(token).decode()
            user_id, timestamp, expires_hours = decoded.split(':')
            
            # Check if token expired
            token_time = datetime.fromisoformat(timestamp)
            expiry_time = token_time + timedelta(hours=int(expires_hours))
            
            if datetime.utcnow() > expiry_time:
                self.logger.warning("API token expired")
                return None
                
            return {'user_id': user_id}
            
        except Exception as e:
            self.logger.warning(f"Invalid API token: {str(e)}")
            return None
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_password(self, password, hashed_password):
        """Validate password against hash"""
        return self.hash_password(password) == hashed_password
