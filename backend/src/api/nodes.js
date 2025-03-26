const express = require('express');
const router = express.Router();
const { validateRequest } = require('../middleware/validation');
const { metricsMiddleware } = require('../middleware/metrics');

router.post('/edge/register', validateRequest, async (req, res) => {
    try {
        const { nodeId, location, capacity } = req.body;
        
        // Register the edge node
        await global.loadBalancer.edge_manager.register_edge_node(nodeId, location, capacity);
        
        res.json({ 
            success: true,
            message: 'Edge node registered successfully'
        });
    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

router.get('/edge/status', async (req, res) => {
    try {
        const edgeNodes = global.loadBalancer.edge_manager.edge_nodes;
        res.json({
            success: true,
            nodes: Object.values(edgeNodes).map(node => ({
                id: node.node_id,
                location: node.location,
                latency: node.latency,
                taskCount: node.task_count,
                capacity: node.capacity
            }))
        });
    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

module.exports = router;
