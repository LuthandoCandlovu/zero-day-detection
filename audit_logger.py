import json
import os
from datetime import datetime
from logger import setup_logger

class AuditLogger:
    """Dedicated logger for security and compliance auditing"""
    
    def __init__(self, audit_log_path='logs/audit.log'):
        self.logger = setup_logger('audit')
        self.audit_log_path = audit_log_path
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
    
    def log_security_event(self, event_type, user, details, severity='INFO'):
        """Log security-related events"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': f"SECURITY_{event_type}",
            'user': user,
            'severity': severity,
            'details': details
        }
        
        # Write to audit log file
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        # Also log through standard logger
        getattr(self.logger, severity.lower())(f"Security event: {event_type}", extra=audit_entry)
    
    def log_data_access(self, user, data_type, access_type, record_count=None):
        """Log data access events for compliance"""
        details = {
            'data_type': data_type,
            'access_type': access_type,
            'record_count': record_count
        }
        
        self.log_security_event('DATA_ACCESS', user, details, 'INFO')
    
    def log_model_access(self, user, model_action, model_version=None):
        """Log model access and modification events"""
        details = {
            'model_action': model_action,
            'model_version': model_version
        }
        
        self.log_security_event('MODEL_ACCESS', user, details, 'INFO')