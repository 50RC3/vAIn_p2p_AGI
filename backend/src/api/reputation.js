const express = require('express');
const router = express.Router();
const { ethers } = require('ethers');
const rateLimit = require('express-rate-limit');
const ReputationContract = require('../../contracts/reputationContract.json');
const ipfs = require('../services/ipfs');

// Initialize provider
const provider = new ethers.providers.JsonRpcProvider(process.env.ETH_RPC_URL);

// Rate limiting
const rateLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // 100 requests per window
});

// Validation middleware
const validateAddress = (req, res, next) => {
    const { address } = req.params;
    if (!ethers.utils.isAddress(address)) {
        return res.status(400).json({ error: 'Invalid Ethereum address' });
    }
    next();
};

router.get('/score/:address', rateLimiter, validateAddress, async (req, res) => {
    try {
        const { address } = req.params;
        const contract = new ethers.Contract(
            process.env.REPUTATION_CONTRACT_ADDRESS,
            ReputationContract.abi,
            provider
        );

        // Add timeout for contract call
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), 5000);
        });

        const score = await Promise.race([
            contract.reputation(address),
            timeoutPromise
        ]);

        // Cache headers for optimizing repeat requests
        res.set('Cache-Control', 'public, max-age=60'); // Cache for 1 minute
        res.json({
            score: score.toString(),
            address,
            timestamp: Date.now()
        });

    } catch (error) {
        console.error(`Reputation error for ${req.params.address}:`, error);
        
        if (error.message === 'Request timeout') {
            return res.status(504).json({ error: 'Request timed out' });
        }
        
        if (error.code === 'NETWORK_ERROR') {
            return res.status(503).json({ error: 'Network error, please try again' });
        }
        
        res.status(500).json({ 
            error: 'Internal server error',
            timestamp: Date.now()
        });
    }
});

router.post('/pheromone', async (req, res) => {
    try {
        const { nodeId, marker } = req.body;
        
        // Validate marker
        if (!marker || !marker.type || typeof marker.strength !== 'number') {
            return res.status(400).json({ error: 'Invalid marker format' });
        }

        const node = await Node.findById(nodeId);
        if (!node) {
            return res.status(404).json({ error: 'Node not found' });
        }

        // Store pheromone data in IPFS
        const ipfsCID = await ipfs.store({
            marker,
            timestamp: Date.now(),
            nodeId
        });

        // Update pheromone marker with IPFS reference
        marker.ipfsCID = ipfsCID;
        node.updatePheromone(marker);
        await node.save();

        // Find nearby nodes and apply diffusion
        const nearbyNodes = await Node.findActive();
        const diffusionPromises = nearbyNodes.map(async nearNode => {
            if (nearNode.id !== nodeId) {
                const diffusedMarker = {
                    ...marker,
                    strength: marker.strength * (marker.diffusionRate || 0.05)
                };
                nearNode.updatePheromone(diffusedMarker);
                return nearNode.save();
            }
        });

        await Promise.all(diffusionPromises);

        // Broadcast updates
        global.wss.clients.forEach(client => {
            if (client.nodeId && nearbyNodes.find(n => n.id === client.nodeId)) {
                client.send(JSON.stringify({
                    type: 'pheromone:update',
                    data: { nodeId, marker }
                }));
            }
        });

        res.json({ success: true, ipfsCID });

    } catch (error) {
        console.error('Pheromone update error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = router;
