const express = require('express');
const router = express.Router();
const ipfsService = require('../services/ipfs');
const logger = require('../utils/logger');
const { createEndpointLimiter } = require('../middleware/rateLimiter');
const config = require('../config');

// Create rate limiters for IPFS endpoints
const ipfsLimiter = createEndpointLimiter(30, 60); // 30 requests per minute

// Helper function to check if a feature is enabled
const featureEnabled = (feature) => {
    return config.ipfs.features && config.ipfs.features[feature] === true;
};

// Get IPFS status with health information
router.get('/status', ipfsLimiter, async (req, res) => {
    try {
        const status = await ipfsService.getStatus();
        const health = ipfsService.getHealthSummary();
        
        res.json({
            success: true,
            status: {
                ...status,
                health
            }
        });
    } catch (err) {
        logger.error('Error retrieving IPFS status', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve IPFS status'
        });
    }
});

// Get IPFS health check history
router.get('/health', ipfsLimiter, async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 20;
        const history = ipfsService.getHealthHistory(limit);
        const summary = ipfsService.getHealthSummary(
            parseInt(req.query.minutes) || 10
        );
        
        res.json({
            success: true,
            summary,
            history
        });
    } catch (err) {
        logger.error('Error retrieving IPFS health', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve IPFS health information'
        });
    }
});

// Force a health check now
router.post('/check', ipfsLimiter, async (req, res) => {
    try {
        const result = await ipfsService.checkHealth();
        res.json({
            success: true,
            result
        });
    } catch (err) {
        logger.error('Error performing IPFS health check', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to perform health check'
        });
    }
});

// Run cleanup based on options in the request body
router.post('/cleanup', ipfsLimiter, async (req, res) => {
    try {
        const options = {
            olderThan: req.body.olderThan || 30 * 24 * 60 * 60 * 1000,
            dryRun: req.body.dryRun !== false,
            keepMetadata: req.body.keepMetadata === true
        };
        
        const result = await ipfsService.cleanup(options);
        res.json({
            success: true,
            result
        });
    } catch (err) {
        logger.error('Error cleaning up IPFS pins', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to clean up IPFS pins'
        });
    }
});

// Get advanced feature status
router.get('/features', ipfsLimiter, async (req, res) => {
    try {
        const featureStatus = ipfsService.getFeatureStatus();
        
        res.json({
            success: true,
            features: featureStatus
        });
    } catch (err) {
        logger.error('Error retrieving IPFS feature status', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve IPFS feature status'
        });
    }
});

// Get list of pins with filtering
router.get('/pins', ipfsLimiter, async (req, res) => {
    if (!featureEnabled('advancedPinning')) {
        return res.status(404).json({
            success: false,
            error: 'Advanced pinning feature is not enabled'
        });
    }
    
    try {
        const options = {
            type: req.query.type || 'all',
            before: req.query.before ? parseInt(req.query.before) : undefined,
            after: req.query.after ? parseInt(req.query.after) : undefined
        };
        
        // Parse metadata filter if provided
        if (req.query.metadata) {
            try {
                options.metadata = JSON.parse(req.query.metadata);
            } catch (err) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid metadata filter format'
                });
            }
        }
        
        const pins = await ipfsService.listPins(options);
        res.json({
            success: true,
            pins
        });
    } catch (err) {
        logger.error('Error listing IPFS pins', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to list IPFS pins'
        });
    }
});

// Pin a CID
router.post('/pins', ipfsLimiter, async (req, res) => {
    if (!featureEnabled('advancedPinning')) {
        return res.status(404).json({
            success: false,
            error: 'Advanced pinning feature is not enabled'
        });
    }
    
    try {
        const { cid, options = {} } = req.body;
        
        if (!cid) {
            return res.status(400).json({
                success: false,
                error: 'CID is required'
            });
        }
        
        const result = await ipfsService.pinCID(cid, options);
        res.json({
            success: true,
            pinned: result
        });
    } catch (err) {
        logger.error('Error pinning CID', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to pin CID'
        });
    }
});

// Check pin status
router.get('/pins/:cid', ipfsLimiter, async (req, res) => {
    if (!featureEnabled('advancedPinning')) {
        return res.status(404).json({
            success: false,
            error: 'Advanced pinning feature is not enabled'
        });
    }
    
    try {
        const { cid } = req.params;
        const type = req.query.type || 'all';
        
        const status = await ipfsService.isPinned(cid, type);
        res.json({
            success: true,
            status
        });
    } catch (err) {
        logger.error('Error checking pin status', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to check pin status'
        });
    }
});

// Remove a pin
router.delete('/pins/:cid', ipfsLimiter, async (req, res) => {
    if (!featureEnabled('advancedPinning')) {
        return res.status(404).json({
            success: false,
            error: 'Advanced pinning feature is not enabled'
        });
    }
    
    try {
        const { cid } = req.params;
        const options = {
            dryRun: req.query.dryRun === 'true',
            keepMetadata: req.query.keepMetadata === 'true',
            remote: req.query.remote !== 'false'
        };
        
        // For advanced pinning, we use the cleanup method with a specific filter
        const result = await ipfsService.cleanup({
            filter: (itemCid) => itemCid === cid,
            dryRun: options.dryRun,
            keepMetadata: options.keepMetadata
        });
        
        res.json({
            success: true,
            result
        });
    } catch (err) {
        logger.error('Error removing pin', { error: err.message });
        res.status(500).json({
            success: false,
            error: 'Failed to remove pin'
        });
    }
});

// Multi-address management routes
if (featureEnabled('multiAddressSupport')) {
    // List multi-addresses
    router.get('/multiaddrs', ipfsLimiter, async (req, res) => {
        try {
            const addresses = ipfsService.getMultiAddresses();
            res.json({
                success: true,
                addresses
            });
        } catch (err) {
            logger.error('Error listing multi-addresses', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to list multi-addresses'
            });
        }
    });

    // Add multi-address
    router.post('/multiaddrs', ipfsLimiter, async (req, res) => {
        try {
            const { address } = req.body;
            
            if (!address) {
                return res.status(400).json({
                    success: false,
                    error: 'Address is required'
                });
            }
            
            const result = await ipfsService.addMultiAddress(address);
            res.json({
                success: true,
                added: result
            });
        } catch (err) {
            logger.error('Error adding multi-address', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to add multi-address'
            });
        }
    });

    // Remove multi-address
    router.delete('/multiaddrs', ipfsLimiter, async (req, res) => {
        try {
            const { address } = req.body;
            
            if (!address) {
                return res.status(400).json({
                    success: false,
                    error: 'Address is required'
                });
            }
            
            const result = await ipfsService.removeMultiAddress(address);
            res.json({
                success: true,
                removed: result
            });
        } catch (err) {
            logger.error('Error removing multi-address', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to remove multi-address'
            });
        }
    });
}

// Remote pinning service management routes
if (featureEnabled('advancedPinning')) {
    // List remote pinning services
    router.get('/services', ipfsLimiter, async (req, res) => {
        try {
            const services = ipfsService.getRemotePinningServices();
            res.json({
                success: true,
                services
            });
        } catch (err) {
            logger.error('Error listing remote pinning services', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to list remote pinning services'
            });
        }
    });

    // Add remote pinning service
    router.post('/services', ipfsLimiter, async (req, res) => {
        try {
            const { name, endpoint, key } = req.body;
            
            if (!name || !endpoint || !key) {
                return res.status(400).json({
                    success: false,
                    error: 'Name, endpoint, and key are required'
                });
            }
            
            const result = await ipfsService.addRemotePinningService(name, endpoint, key);
            res.json({
                success: true,
                added: result
            });
        } catch (err) {
            logger.error('Error adding remote pinning service', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to add remote pinning service'
            });
        }
    });

    // Remove remote pinning service
    router.delete('/services/:name', ipfsLimiter, async (req, res) => {
        try {
            const { name } = req.params;
            const result = await ipfsService.removeRemotePinningService(name);
            res.json({
                success: true,
                removed: result
            });
        } catch (err) {
            logger.error('Error removing remote pinning service', { error: err.message });
            res.status(500).json({
                success: false,
                error: 'Failed to remove remote pinning service'
            });
        }
    });
}

module.exports = router;
