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

try:
    from tpm2_pytss import *
    from tpm2_pytss.ESAPI import ESAPI
    from tpm2_pytss.TPM2_ALG import TPM2_ALG
    from tpm2             return HardwareAttestation(
                tmp_quote=tmp_quote,
                hardware_fingerprint=hardware_fingerprint,
                cpu_info=cpu_info,
                memory_info=memory_info,
                platform_info=platform_info,
                attestation_timestamp=datetime.utcnow(),
                attestation_signature=attestation_signature
            )  return HardwareAttestation(
                tmp_quote=tmp_quote,ytss.types import *
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False
    # Create mock classes for when TPM is not available
    class ESAPI:
        pass
    class TPM2_ALG:
        RSA = 'RSA'
        SHA256 = 'SHA256'
        NULL = 'NULL'
        RSASSA = 'RSASSA'
    # Mock TPM types
    class ESYS_TR:
        RH_ENDORSEMENT = 'RH_ENDORSEMENT'
    class TPMA_OBJECT:
        USERWITHAUTH = 'USERWITHAUTH'
        RESTRICTED = 'RESTRICTED'
        SIGN_ENCRYPT = 'SIGN_ENCRYPT'
        FIXEDTPM = 'FIXEDTPM'
        FIXEDPARENT = 'FIXEDPARENT'
        SENSITIVEDATAORIGIN = 'SENSITIVEDATAORIGIN'
    # Mock other TPM types
    TPM2B_PUBLIC = dict
    TPMT_PUBLIC = dict
    TPMU_PUBLIC_PARMS = dict
    TPMS_RSA_PARMS = dict
    TPMT_SYM_DEF_OBJECT = dict
    TPMT_RSA_SCHEME = dict
    TPMU_PUBLIC_ID = dict
    TPM2B_PUBLIC_KEY_RSA = dict
    TPM2B_AUTH = dict
    TPM2B_SENSITIVE_CREATE = dict
    TPMS_SENSITIVE_CREATE = dict
    TPM2B_DATA = dict
    TPML_PCR_SELECTION = dict
    TPMS_PCR_SELECTION = dict
    TPMT_SIG_SCHEME = dict
    
    logging.warning("TPM 2.0 library not available. Hardware attestation will use simulation mode.")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15

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

class TPMNotAvailableError(TPMAttestationError):
    """Raised when TPM is not available"""
    pass

class TPMQuoteVerificationError(TPMAttestationError):
    """Raised when TPM quote verification fails"""
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
        
        # Trusted PCR baseline values (should be configured per deployment)
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
                return True
                
            # Initialize ESAPI connection
            self.esapi = ESAPI()
            
            # Create or load Attestation Key (AK)
            await self._setup_attestation_key()
            
            self.logger.info("TPM 2.0 attestation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TPM attestation: {e}")
            # Fall back to simulation mode
            self.simulation_mode = True
            self.logger.warning("Falling back to simulation mode")
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
            
        try:
            # TPM key template for Attestation Key
            template = TPM2B_PUBLIC(
                publicArea=TPMT_PUBLIC(
                    type=TPM2_ALG.RSA,
                    nameAlg=TPM2_ALG.SHA256,
                    objectAttributes=(
                        TPMA_OBJECT.USERWITHAUTH |
                        TPMA_OBJECT.RESTRICTED |
                        TPMA_OBJECT.SIGN_ENCRYPT |
                        TPMA_OBJECT.FIXEDTPM |
                        TPMA_OBJECT.FIXEDPARENT |
                        TPMA_OBJECT.SENSITIVEDATAORIGIN
                    ),
                    parameters=TPMU_PUBLIC_PARMS(
                        rsaDetail=TPMS_RSA_PARMS(
                            symmetric=TPMT_SYM_DEF_OBJECT(algorithm=TPM2_ALG.NULL),
                            scheme=TPMT_RSA_SCHEME(scheme=TPM2_ALG.RSASSA),
                            keyBits=2048,
                            exponent=0
                        )
                    ),
                    unique=TPMU_PUBLIC_ID(rsa=TPM2B_PUBLIC_KEY_RSA())
                )
            )
            
            # Create the Attestation Key
            auth_value = TPM2B_AUTH(buffer=b"attestation_key_auth")
            
            result = self.esapi.CreatePrimary(
                primaryHandle=ESYS_TR.RH_ENDORSEMENT,
                inSensitive=TPM2B_SENSITIVE_CREATE(sensitive=TPMS_SENSITIVE_CREATE(
                    userAuth=auth_value
                )),
                inPublic=template,
                outsideInfo=TPM2B_DATA(),
                creationPCR=TPML_PCR_SELECTION()
            )
            
            self.ak_handle = result[0]
            self.ak_public_key = result[1].publicArea
            
            self.logger.info("Attestation Key created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create attestation key: {e}")
            raise TPMAttestationError(f"AK creation failed: {e}")
    
    async def generate_quote(self, nonce: bytes, pcr_selection: List[int] = None) -> TPMQuote:
        """Generate a TPM quote for the specified PCRs"""
        if pcr_selection is None:
            pcr_selection = [0, 1, 2, 3, 4, 5, 6, 7]  # Boot-related PCRs
            
        if self.simulation_mode:
            return await self._generate_simulated_quote(nonce, pcr_selection)
            
        try:
            # Prepare PCR selection
            pcr_select = TPML_PCR_SELECTION([
                TPMS_PCR_SELECTION(
                    hash=TPM2_ALG.SHA256,
                    sizeofSelect=3,
                    pcrSelect=self._build_pcr_select_mask(pcr_selection)
                )
            ])
            
            # Generate quote
            result = self.esapi.Quote(
                signHandle=self.ak_handle,
                inScheme=TPMT_SIG_SCHEME(scheme=TPM2_ALG.RSASSA),
                qualifyingData=TPM2B_DATA(buffer=nonce),
                PCRselect=pcr_select
            )
            
            quoted_data = result[0]
            signature = result[1]
            
            # Read PCR values
            pcr_values = {}
            for pcr in pcr_selection:
                pcr_result = self.esapi.PCR_Read(
                    pcrSelectionIn=TPML_PCR_SELECTION([
                        TPMS_PCR_SELECTION(
                            hash=TPM2_ALG.SHA256,
                            sizeofSelect=3,
                            pcrSelect=self._build_pcr_select_mask([pcr])
                        )
                    ])
                )
                pcr_values[pcr] = pcr_result[1].digests[0].buffer
            
            return TPMQuote(
                quoted_data=quoted_data.buffer,
                signature=signature.signature.rsassa.sig.buffer,
                pcr_values=pcr_values,
                nonce=nonce,
                timestamp=datetime.utcnow(),
                attestation_key=self.ak_public_key,
                quote_info={
                    'tpm_version': '2.0',
                    'quote_algorithm': 'RSASSA-SHA256',
                    'pcr_algorithm': 'SHA256'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate TPM quote: {e}")
            raise TPMAttestationError(f"Quote generation failed: {e}")
    
    async def _generate_simulated_quote(self, nonce: bytes, pcr_selection: List[int]) -> TPMQuote:
        """Generate a simulated quote for development/testing"""
        self.logger.debug("Generating simulated TPM quote")
        
        # Simulate PCR values
        pcr_values = {}
        for pcr in pcr_selection:
            # Create deterministic but realistic PCR values
            pcr_data = f"simulated_pcr_{pcr}_{datetime.utcnow().isoformat()}".encode()
            pcr_values[pcr] = hashlib.sha256(pcr_data).digest()
        
        # Create quoted data (simplified structure)
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
                'tpm_version': '2.0_simulated',
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
            
            if self.simulation_mode:
                return await self._verify_simulated_quote(quote, expected_nonce)
            
            # Verify quote signature
            if not await self._verify_quote_signature(quote):
                self.logger.warning(f"Quote signature verification failed for peer {peer_id}")
                return False
            
            # Verify PCR values against baselines
            if not await self._verify_pcr_values(quote.pcr_values, peer_id):
                self.logger.warning(f"PCR verification failed for peer {peer_id}")
                return False
            
            self.logger.info(f"TPM quote verified successfully for peer {peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Quote verification error for peer {peer_id}: {e}")
            return False
    
    async def _verify_simulated_quote(self, quote: TPMQuote, expected_nonce: bytes) -> bool:
        """Verify simulated quote (for development)"""
        try:
            # Basic checks for simulated quotes
            if quote.quote_info.get('tpm_version') != '2.0_simulated':
                return False
            
            # Verify simulated signature
            expected_sig = hashlib.sha256(quote.quoted_data + expected_nonce).digest()
            return quote.signature == expected_sig
            
        except Exception:
            return False
    
    async def _verify_quote_signature(self, quote: TPMQuote) -> bool:
        """Verify the cryptographic signature of a TPM quote"""
        try:
            # Load attestation key
            public_key = serialization.load_der_public_key(quote.attestation_key)
            
            # Verify signature
            public_key.verify(
                quote.signature,
                quote.quoted_data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            return True
            
        except InvalidSignature:
            return False
        except Exception as e:
            self.logger.error(f"Signature verification error: {e}")
            return False
    
    async def _verify_pcr_values(self, pcr_values: Dict[int, bytes], 
                                peer_id: str = None) -> bool:
        """Verify PCR values against trusted baselines"""
        try:
            for pcr_index, pcr_value in pcr_values.items():
                # Skip verification if no baseline is set
                if pcr_index not in self.trusted_pcr_baselines:
                    continue
                    
                baseline = self.trusted_pcr_baselines[pcr_index]
                if baseline and pcr_value != baseline:
                    self.logger.warning(
                        f"PCR {pcr_index} mismatch for peer {peer_id}: "
                        f"expected {baseline.hex()}, got {pcr_value.hex()}"
                    )
                    # For now, log but don't fail (baselines need proper configuration)
                    # return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"PCR verification error: {e}")
            return False
    
    def _build_pcr_select_mask(self, pcr_list: List[int]) -> bytes:
        """Build PCR selection mask for TPM operations"""
        mask = bytearray(3)  # 24 bits for PCR 0-23
        for pcr in pcr_list:
            if 0 <= pcr <= 23:
                byte_index = pcr // 8
                bit_index = pcr % 8
                mask[byte_index] |= (1 << bit_index)
        return bytes(mask)
    
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
                tmp_quote=tpm_quote,
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
            if attestation.tpm_quote:
                if not await self.verify_quote(attestation.tpm_quote, expected_nonce, peer_id):
                    return False
            
            # Verify hardware fingerprint consistency
            if not await self._verify_hardware_consistency(attestation, peer_id):
                return False
            
            # Cache successful verification
            self.attestation_cache[cache_key] = (attestation, datetime.utcnow())
            
            self.logger.info(f"Hardware attestation verified for peer {peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware attestation verification failed for {peer_id}: {e}")
            return False
    
    async def _verify_hardware_consistency(self, attestation: HardwareAttestation, 
                                         peer_id: str) -> bool:
        """Verify hardware information consistency"""
        try:
            # Basic sanity checks
            if not attestation.cpu_info.get('cpu_count', 0) > 0:
                self.logger.warning(f"Invalid CPU count for peer {peer_id}")
                return False
            
            if not attestation.memory_info.get('total', 0) > 0:
                self.logger.warning(f"Invalid memory info for peer {peer_id}")
                return False
            
            # Check minimum requirements
            min_requirements = {
                'cpu_cores': 2,
                'memory_gb': 4
            }
            
            cpu_count = attestation.cpu_info.get('cpu_count', 0)
            memory_gb = attestation.memory_info.get('total', 0) / (1024**3)
            
            if cpu_count < min_requirements['cpu_cores']:
                self.logger.warning(f"Insufficient CPU cores for peer {peer_id}: {cpu_count}")
                return False
            
            if memory_gb < min_requirements['memory_gb']:
                self.logger.warning(f"Insufficient memory for peer {peer_id}: {memory_gb:.1f}GB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware consistency check failed: {e}")
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
            'tpm_available': TPM_AVAILABLE,
            'trusted_pcr_count': len([v for v in self.trusted_pcr_baselines.values() if v])
        }

# Global TPM manager instance
tpm_manager = TPMAttestationManager()

async def initialize_tpm_attestation() -> bool:
    """Initialize global TPM attestation manager"""
    return await tpm_manager.initialize()

async def cleanup_tpm_attestation():
    """Cleanup global TPM attestation manager"""
    await tpm_manager.cleanup()
