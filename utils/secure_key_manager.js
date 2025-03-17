const { SecureEnclave } = require('@secure-enclave/node');
const { HSM } = require('@hsm/client');

class SecureKeyManager {
    constructor(config) {
        this.useHSM = config.useHSM;
        this.useEnclave = config.useEnclave;
        
        if (this.useHSM) {
            this.hsm = new HSM(config.hsmConfig);
        }
        
        if (this.useEnclave) {
            this.enclave = new SecureEnclave();
        }
    }
    
    async generateKeyPair() {
        if (this.useHSM) {
            return await this.hsm.generateKeyPair();
        } else if (this.useEnclave) {
            return await this.enclave.generateKeyPair();
        }
    }
    
    async sign(message, keyId) {
        if (this.useHSM) {
            return await this.hsm.sign(message, keyId);
        } else if (this.useEnclave) {
            return await this.enclave.sign(message, keyId);
        }
    }
}

module.exports = SecureKeyManager;
