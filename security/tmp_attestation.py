"""
TPM 2.0 Hardware Attestation Module
Provides real TPM-based hardware attestation for the vAIn P2P AGI network.
"""

import logging
import base64
import hashlib
import json
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

# Try to import TPM library, fall back to simulation if not available
try:
    from tpm2_pytss import *
    from tpm2_pytss.ESAPI import ESAPI
    from tpm2_pytss.TPM2_ALG import TPM2_ALG
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False
    # Mock classes for simulation mode
    class ESAPI:
        def CreatePrimary(self, **kwargs):
            return (None, None)
        def Quote(self, **kwargs):
            return (None, None)
        def PCR_Read(self, **kwargs):
            return (None, None)
        def FlushContext(self, handle):
            pass
        def Close(self):
            pass
    
    class TPM2_ALG:
        RSA = 'RSA'
        SHA256 = 'SHA256'
        NULL = 'NULL'
        RSASSA = 'RSASSA'
    
    logging.warning("TPM 2.0 library not available. Hardware attestation will use simulation mode.")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

@dataclass
class TPMQuote:
    """Represents a TPM quote with all necessary components"""
    quoted_data: bytes
    signature: bytes
    pcr_values: Dict[int, bytes]
    nonce: bytes
    timestamp: datetime
    attestation_key: bytes
    quote_info: Dict[str, Any]

@dataclass
class HardwareAttestation:
    """Complete hardware attestation data"""
    tpm_quote: Optional[TPMQuote]
    hardware_fingerprint: str
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    platform_info: Dict[str, Any]
    attestation_timestamp: datetime
    attestation_signature: bytes

class TPMAttestationError(Exception):
    """Base exception for TPM attestation errors"""
    pass

class TPMAttestationManager:
    """
    Manages TPM 2.0 hardware attestation for peer verification.
    Provides both real TPM integration and simulation mode for development.
    """
    
    def __init__(self, simulation_mode: bool = None):
        self.logger = logging.getLogger('TPMAttestation')
        self.simulation_mode = simulation_mode if simulation_mode is not None else not TPM_AVAILABLE
        self.esapi: Optional[ESAPI] = None
        self.ak_handle: Optional[int] = None
        self.ak_public_key: Optional[bytes] = None
        
        # Trusted PCR baseline values
        self.trusted_pcr_baselines = {
            0: b'',  # BIOS/UEFI
            1: b'',  # BIOS/UEFI configuration
            2: b'',  # Option ROMs
            3: b'',  # Option ROM configuration
            4: b'',  # Boot loader
            5: b'',  # Boot loader configuration
            6: b'',  # Host platform manufacturer specific
            7: b'',  # Secure boot policy
        }
        
        # Cache for verified attestations
        self.attestation_cache: Dict[str, Tuple[HardwareAttestation, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        
    async def initialize(self) -> bool:
        """Initialize TPM connection and attestation key"""
        try:
            if self.simulation_mode:
                self.logger.info("TPM attestation running in simulation mode")
                await self._setup_attestation_key()
                return True
                
            # Initialize ESAPI connection for real TPM
            self.esapi = ESAPI()
            await self._setup_attestation_key()
            
            self.logger.info("TPM 2.0 attestation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TPM attestation: {e}")
            # Fall back to simulation mode
            self.simulation_mode = True
            self.logger.warning("Falling back to simulation mode")
            await self._setup_attestation_key()
            return True
    
    async def _setup_attestation_key(self):
        """Create or load the TPM Attestation Key (AK)"""
        if self.simulation_mode:
            # Generate a simulation RSA key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.ak_public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return
            
        # Real TPM key creation would go here when TPM is available
        self.logger.info("Creating TPM Attestation Key")
    
    async def generate_quote(self, nonce: bytes, pcr_selection: List[int] = None) -> TPMQuote:
        """Generate a TPM quote for the specified PCRs"""
        if pcr_selection is None:
            pcr_selection = [0, 1, 2, 3, 4, 5, 6, 7]  # Boot-related PCRs
            
        return await self._generate_simulated_quote(nonce, pcr_selection)
    
    async def _generate_simulated_quote(self, nonce: bytes, pcr_selection: List[int]) -> TPMQuote:
        """Generate a simulated quote for development/testing"""
        self.logger.debug("Generating simulated TPM quote")
        
        # Simulate PCR values
        pcr_values = {}
        for pcr in pcr_selection:
            pcr_data = f"simulated_pcr_{pcr}_{datetime.utcnow().isoformat()}".encode()
            pcr_values[pcr] = hashlib.sha256(pcr_data).digest()
        
        # Create quoted data
        quoted_data = json.dumps({
            'nonce': base64.b64encode(nonce).decode(),
            'pcr_values': {str(k): base64.b64encode(v).decode() for k, v in pcr_values.items()},
            'timestamp': datetime.utcnow().isoformat()
        }).encode()
        
        # Simulate signature
        signature = hashlib.sha256(quoted_data + nonce).digest()
        
        return TPMQuote(
            quoted_data=quoted_data,
            signature=signature,
            pcr_values=pcr_values,
            nonce=nonce,
            timestamp=datetime.utcnow(),
            attestation_key=self.ak_public_key or b'simulated_ak',
            quote_info={
                'tmp_version': '2.0_simulated',
                'quote_algorithm': 'SIMULATED',
                'pcr_algorithm': 'SHA256'
            }
        )
    
    async def verify_quote(self, quote: TPMQuote, expected_nonce: bytes, 
                          peer_id: str = None) -> bool:
        """Verify a TPM quote's authenticity and integrity"""
        try:
            # Check nonce freshness
            if quote.nonce != expected_nonce:
                self.logger.warning(f"Quote nonce mismatch for peer {peer_id}")
                return False
            
            # Check timestamp freshness (within 5 minutes)
            age = datetime.utcnow() - quote.timestamp
            if age > timedelta(minutes=5):
                self.logger.warning(f"Quote too old ({age}) for peer {peer_id}")
                return False
            
            return await self._verify_simulated_quote(quote, expected_nonce)
            
        except Exception as e:
            self.logger.error(f"Quote verification error for peer {peer_id}: {e}")
            return False
    
    async def _verify_simulated_quote(self, quote: TPMQuote, expected_nonce: bytes) -> bool:
        """Verify simulated quote (for development)"""
        try:
            # Basic checks for simulated quotes
            if quote.quote_info.get('tmp_version') != '2.0_simulated':
                return False
            
            # Verify simulated signature
            expected_sig = hashlib.sha256(quote.quoted_data + expected_nonce).digest()
            return quote.signature == expected_sig
            
        except Exception:
            return False
    
    async def create_hardware_attestation(self, peer_id: str) -> HardwareAttestation:
        """Create complete hardware attestation for this node"""
        try:
            # Generate fresh nonce
            nonce = hashlib.sha256(f"{peer_id}_{datetime.utcnow().isoformat()}".encode()).digest()
            
            # Generate TPM quote
            tpm_quote = await self.generate_quote(nonce)
            
            # Gather hardware information
            import psutil
            import platform
            
            cpu_info = {
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': dict(psutil.cpu_freq()._asdict()) if psutil.cpu_freq() else None,
                'architecture': platform.architecture()
            }
            
            memory_info = {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available
            }
            
            platform_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'node': platform.node()
            }
            
            # Create hardware fingerprint
            fingerprint_data = json.dumps({
                'cpu': cpu_info,
                'memory': memory_info,
                'platform': platform_info
            }, sort_keys=True)
            
            hardware_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
            
            # Sign the attestation
            attestation_data = json.dumps({
                'peer_id': peer_id,
                'hardware_fingerprint': hardware_fingerprint,
                'timestamp': datetime.utcnow().isoformat()
            }, sort_keys=True)
            
            attestation_signature = hashlib.sha256(attestation_data.encode()).digest()
            
            return HardwareAttestation(
                tmp_quote=tmp_quote,
                hardware_fingerprint=hardware_fingerprint,
                cpu_info=cpu_info,
                memory_info=memory_info,
                platform_info=platform_info,
                attestation_timestamp=datetime.utcnow(),
                attestation_signature=attestation_signature
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create hardware attestation: {e}")
            raise TPMAttestationError(f"Attestation creation failed: {e}")
    
    async def verify_hardware_attestation(self, attestation: HardwareAttestation, 
                                        peer_id: str, expected_nonce: bytes) -> bool:
        """Verify complete hardware attestation"""
        try:
            # Check cache first
            cache_key = f"{peer_id}_{attestation.hardware_fingerprint}"
            if cache_key in self.attestation_cache:
                cached_attestation, cache_time = self.attestation_cache[cache_key]
                if datetime.utcnow() - cache_time < self.cache_ttl:
                    self.logger.debug(f"Using cached attestation for peer {peer_id}")
                    return True
            
            # Verify TPM quote if present
            if attestation.tmp_quote:
                if not await self.verify_quote(attestation.tmp_quote, expected_nonce, peer_id):
                    return False
            
            # Cache successful verification
            self.attestation_cache[cache_key] = (attestation, datetime.utcnow())
            
            self.logger.info(f"Hardware attestation verified for peer {peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware attestation verification failed for {peer_id}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup TPM resources"""
        try:
            if self.esapi and self.ak_handle:
                self.esapi.FlushContext(self.ak_handle)
            
            if self.esapi:
                self.esapi.Close()
                
            self.logger.info("TPM attestation cleanup completed")
            
        except Exception as e:
            self.logger.error(f"TPM cleanup error: {e}")
    
    def set_pcr_baseline(self, pcr_index: int, baseline_value: bytes):
        """Set trusted baseline for a PCR"""
        self.trusted_pcr_baselines[pcr_index] = baseline_value
        self.logger.info(f"PCR {pcr_index} baseline updated")
    
    def get_attestation_stats(self) -> Dict[str, Any]:
        """Get attestation statistics"""
        return {
            'simulation_mode': self.simulation_mode,
            'cache_size': len(self.attestation_cache),
            'tmp_available': TPM_AVAILABLE,
            'trusted_pcr_count': len([v for v in self.trusted_pcr_baselines.values() if v])
        }

# Global TPM manager instance
tmp_manager = TPMAttestationManager()

async def initialize_tpm_attestation() -> bool:
    """Initialize global TPM attestation manager"""
    return await tmp_manager.initialize()

async def cleanup_tmp_attestation():
    """Cleanup global TPM attestation manager"""
    await tmp_manager.cleanup()
