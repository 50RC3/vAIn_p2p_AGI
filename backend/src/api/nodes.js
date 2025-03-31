const express = require('express');
const router = express.Router();
const { validateRequest } = require('../middleware/validation');
const { metricsMiddleware } = require('../middleware/metrics');
const nodeService = require('../services/nodeService');
const logger = require('../utils/logger');

// Register a new edge node
router.post('/edge/register', validateRequest, async (req, res) => {
    try {
        const { nodeId, location, capacity } = req.body;
        
        // Register the edge node using our service
        const node = await nodeService.registerNode({
            nodeId,
            location,
            capacity,
            status: 'inactive' // Default status
        });
        
        res.json({ 
            success: true,
            message: 'Edge node registered successfully',
            node: {
                nodeId: node.nodeId,
                status: node.status
            }
        });
    } catch (err) {
        logger.error('Edge node registration failed', { error: err.message });
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

// Get status of all edge nodes
router.get('/edge/status', async (req, res) => {
    try {
        const nodes = await nodeService.listActiveNodes();
        
        res.json({
            success: true,
            nodes: nodes.map(node => ({
                id: node.nodeId,
                status: node.status,
                lastHeartbeat: node.lastHeartbeat,
                metrics: node.metrics
            }))
        });
    } catch (err) {
        logger.error('Edge node status retrieval failed', { error: err.message });
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

// Update node metrics
router.post('/edge/:nodeId/metrics', validateRequest, async (req, res) => {
    try {
        const { nodeId } = req.params;
        const metrics = req.body.metrics;
        
        if (!metrics) {
            return res.status(400).json({
                success: false,
                error: 'Metrics data is required'
            });
        }
        
        const node = await nodeService.updateNodeMetrics(nodeId, metrics);
        
        res.json({
            success: true,
            message: 'Node metrics updated successfully'
        });
    } catch (err) {
        logger.error('Node metrics update failed', { 
            nodeId: req.params.nodeId,
            error: err.message 
        });
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

// Get node health
router.get('/edge/:nodeId/health', async (req, res) => {
    try {
        const { nodeId } = req.params;
        const health = await nodeService.getNodeHealth(nodeId);
        
        res.json({
            success: true,
            health
        });
    } catch (err) {
        logger.error('Node health check failed', { 
            nodeId: req.params.nodeId,
            error: err.message 
        });
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

module.exports = router;
