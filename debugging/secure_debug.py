import logging
import hashlib
import os
import time
import socket
import json
import asyncio
import hmac
import base64
import secrets
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import ipaddress
from datetime import datetime, timedelta

from utils.unified_logger import get_logger
from debugging.debug_config import DebugConfigManager

logger = get_logger("secure_debug")

@dataclass
class DebugSession:
    """Debug session with security credentials"""
    session_id: str
    token: str
    ip_address: str
    created: float = field(default_factory=time.time)
    expires: float = field(default_factory=lambda: time.time() + 3600)  # 1-hour default
    permissions: List[str] = field(default_factory=lambda: ["view"])

@dataclass
class DebugAccessRecord:
    """Record of a debug access attempt"""
    timestamp: float = field(default_factory=time.time)
    ip_address: str = ""
    endpoint: str = ""
    method: str = "GET"
    user_agent: str = ""
    session_id: str = ""
    success: bool = False
    failure_reason: str = ""

class SecureDebugManager:
    """Manages secure access to debugging facilities"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "SecureDebugManager":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = SecureDebugManager()
        return cls._instance
    
    def __init__(self):
        """Initialize secure debug manager"""
        self.config_manager = DebugConfigManager.get_instance()
        self.sessions: Dict[str, DebugSession] = {}
        self.access_log: List[DebugAccessRecord] = []
        self.max_access_log = 1000
        self.failed_attempts: Dict[str, List[float]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # seconds
        self._secret_key = self._load_or_generate_secret()
        
    def _load_or_generate_secret(self) -> bytes:
        """Load existing secret key or generate a new one"""
        secret_path = os.path.join("config", ".debug_secret")
        
        try:
            if os.path.exists(secret_path):
                with open(secret_path, "rb") as f:
                    secret = f.read()
                    if len(secret) >= 32:
                        return secret
        except Exception:
            # If reading fails, generate a new key
            pass
            
        # Generate new secret key
        secret = secrets.token_bytes(64)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(secret_path), exist_ok=True)
            # Save with restricted permissions
            with open(secret_path, "wb") as f:
                f.write(secret)
            os.chmod(secret_path, 0o600)  # Read/write for owner only
        except Exception as e:
            logger.error(f"Failed to save debug secret key: {e}", exc_info=True)
            
        return secret
        
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if an IP address is allowed to connect"""
        if not self.config_manager.config.require_auth:
            return True
            
        # Check if IP is in allowed list
        allowed_ips = self.config_manager.config.allowed_ips
        
        # Special case: if allowed_ips contains "0.0.0.0", allow all
        if "0.0.0.0" in allowed_ips:
            return True
            
        # Check for exact match
        if ip_address in allowed_ips:
            return True
            
        # Check for subnet matches (e.g., 192.168.1.0/24)
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            for allowed in allowed_ips:
                if "/" in allowed:  # CIDR notation
                    network = ipaddress.ip_network(allowed, strict=False)
                    if ip_obj in network:
                        return True
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip_address}")
            
        return False
        
    def create_session(self, ip_address: str, permissions: List[str] = None) -> Optional[DebugSession]:
        """Create a new debug session if IP is allowed"""
        if not self.is_ip_allowed(ip_address):
            logger.warning(f"Debug session creation denied for unauthorized IP: {ip_address}")
            return None
            
        # Check for lockout
        if self._is_ip_locked_out(ip_address):
            logger.warning(f"Debug session denied for locked out IP: {ip_address}")
            return None
            
        # Generate session ID and token
        session_id = secrets.token_urlsafe(16)
        token = secrets.token_urlsafe(32)
        
        # Set default permissions if none provided
        if permissions is None:
            permissions = ["view"]
            
        # Create session with one-hour expiration
        session = DebugSession(
            session_id=session_id,
            token=token,
            ip_address=ip_address,
            created=time.time(),
            expires=time.time() + 3600,
            permissions=permissions
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created debug session {session_id} for {ip_address}")
        
        # Clean up expired sessions occasionally
        if len(self.sessions) % 10 == 0:
            self._cleanup_expired_sessions()
            
        return session
        
    def validate_session(self, session_id: str, token: str, ip_address: str) -> bool:
        """Validate a debug session"""
        session = self.sessions.get(session_id)
        
        if not session:
            self._log_failed_attempt(ip_address, "Invalid session ID")
            return False
            
        # Check if token matches
        if not hmac.compare_digest(session.token, token):
            self._log_failed_attempt(ip_address, "Invalid token")
            return False
            
        # Check if session is expired
        if time.time() > session.expires:
            self._log_failed_attempt(ip_address, "Expired session")
            del self.sessions[session_id]
            return False
            
        # Check if IP matches
        if session.ip_address != ip_address:
            self._log_failed_attempt(ip_address, "IP mismatch")
            return False
            
        # Update session expiration
        session.expires = time.time() + 3600  # Extend by one hour
        
        return True
        
    def _is_ip_locked_out(self, ip_address: str) -> bool:
        """Check if IP is locked out due to failed attempts"""
        if ip_address not in self.failed_attempts:
            return False
            
        attempts = self.failed_attempts[ip_address]
        
        # Remove attempts older than lockout duration
        now = time.time()
        recent_attempts = [t for t in attempts if now - t < self.lockout_duration]
        self.failed_attempts[ip_address] = recent_attempts
        
        # If still over threshold after cleanup, IP is locked out
        return len(recent_attempts) >= self.max_failed_attempts
        
    def _log_failed_attempt(self, ip_address: str, reason: str) -> None:
        """Log a failed authentication attempt"""
        logger.warning(f"Failed debug authentication from {ip_address}: {reason}")
        
        # Record the attempt
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []
            
        self.failed_attempts[ip_address].append(time.time())
        
    def log_access(self, record: DebugAccessRecord) -> None:
        """Log an access attempt"""
        self.access_log.append(record)
        
        # Trim log if needed
        if len(self.access_log) > self.max_access_log:
            self.access_log = self.access_log[-self.max_access_log:]
            
    def get_access_logs(self, limit: int = 100) -> List[DebugAccessRecord]:
        """Get recent access logs"""
        return sorted(self.access_log, key=lambda r: r.timestamp, reverse=True)[:limit]
        
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions"""
        now = time.time()
        expired = [sid for sid, session in self.sessions.items() if session.expires < now]
        for sid in expired:
            del self.sessions[sid]
            
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired debug sessions")
            
    def check_permission(self, session_id: str, permission: str) -> bool:
        """Check if a session has a specific permission"""
        session = self.sessions.get(session_id)
        if not session:
            return False
            
        # Special "admin" permission includes all permissions
        if "admin" in session.permissions:
            return True
            
        return permission in session.permissions
        
    def generate_signed_token(self, data: Dict[str, Any]) -> str:
        """Generate a signed token containing data"""
        # Add expiration
        data["exp"] = int(time.time()) + 3600
        
        # Serialize and encode data
        serialized = json.dumps(data).encode('utf-8')
        data_b64 = base64.urlsafe_b64encode(serialized).decode('utf-8')
        
        # Create signature
        signature = hmac.new(
            self._secret_key,
            data_b64.encode('utf-8'),
            hashlib.sha256
        ).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).decode('utf-8')
        
        # Combine data and signature
        return f"{data_b64}.{sig_b64}"
        
    def verify_signed_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a signed token and return its data"""
        try:
            # Split token into data and signature parts
            parts = token.split(".")
            if len(parts) != 2:
                return None
                
            data_b64, sig_b64 = parts
            
            # Verify signature
            signature = base64.urlsafe_b64decode(sig_b64)
            expected_sig = hmac.new(
                self._secret_key,
                data_b64.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(signature, expected_sig):
                return None
                
            # Decode data
            data_bytes = base64.urlsafe_b64decode(data_b64)
            data = json.loads(data_bytes.decode('utf-8'))
            
            # Check expiration
            if "exp" in data and data["exp"] < int(time.time()):
                return None
                
            return data
            
        except Exception:
            return None

secure_debug_manager = SecureDebugManager.get_instance()