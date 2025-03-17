import jwt
from typing import Dict, Optional
from datetime import datetime, timedelta

class NodeAuthenticator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        
    def generate_token(self, node_id: str) -> str:
        payload = {
            'node_id': node_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None
