const Node = require('../models/node');
const ipfs = require('./ipfs');
const logger = require('../utils/logger');

class NodeService {
    /**
     * Register a new node or update an existing one
     * @param {Object} nodeData - Node data to register
     * @returns {Promise<Object>} The registered node
     */
    async registerNode(nodeData) {
        try {
            const { nodeId } = nodeData;
            
            // Check if node already exists
            let node = await Node.findOne({ nodeId });
            
            if (node) {
                // Update existing node
                Object.assign(node, nodeData);
                logger.info(`Node updated: ${nodeId}`);
            } else {
                // Create new node
                node = new Node(nodeData);
                logger.info(`New node registered: ${nodeId}`);
            }
            
            await node.save();
            return node;
        } catch (error) {
            logger.error('Failed to register node', { error, nodeData });
            throw new Error(`Node registration failed: ${error.message}`);
        }
    }
    
    /**
     * Update node metrics
     * @param {String} nodeId - ID of the node to update
     * @param {Object} metrics - Metrics data to update
     * @returns {Promise<Object>} The updated node
     */
    async updateNodeMetrics(nodeId, metrics) {
        try {
            const node = await Node.findOne({ nodeId });
            
            if (!node) {
                throw new Error(`Node not found: ${nodeId}`);
            }
            
            await node.updateMetrics(metrics);
            logger.debug(`Metrics updated for node: ${nodeId}`);
            
            return node;
        } catch (error) {
            logger.error('Failed to update node metrics', { error, nodeId });
            throw new Error(`Metrics update failed: ${error.message}`);
        }
    }
    
    /**
     * Store node data in IPFS
     * @param {String} nodeId - ID of the node
     * @param {Object} data - Data to store in IPFS
     * @returns {Promise<String>} CID of the stored data
     */
    async storeNodeDataInIPFS(nodeId, data) {
        try {
            const node = await Node.findOne({ nodeId });
            
            if (!node) {
                throw new Error(`Node not found: ${nodeId}`);
            }
            
            const cid = await ipfs.store(data);
            
            node.ipfsData = {
                ...node.ipfsData,
                metadataCID: cid,
                lastUpdate: new Date()
            };
            
            await node.save();
            logger.info(`Node data stored in IPFS`, { nodeId, cid });
            
            return cid;
        } catch (error) {
            logger.error('Failed to store node data in IPFS', { error, nodeId });
            throw new Error(`IPFS storage failed: ${error.message}`);
        }
    }
    
    /**
     * Get node health information
     * @param {String} nodeId - ID of the node
     * @returns {Promise<Object>} Health information
     */
    async getNodeHealth(nodeId) {
        try {
            const node = await Node.findOne({ nodeId });
            
            if (!node) {
                throw new Error(`Node not found: ${nodeId}`);
            }
            
            const health = node.checkHealth();
            logger.debug(`Retrieved health for node: ${nodeId}`);
            
            return health;
        } catch (error) {
            logger.error('Failed to get node health', { error, nodeId });
            throw new Error(`Health check failed: ${error.message}`);
        }
    }
    
    /**
     * List all active nodes
     * @returns {Promise<Array>} List of active nodes
     */
    async listActiveNodes() {
        try {
            const nodes = await Node.find({ status: 'active' })
                .select('nodeId status stake metrics lastHeartbeat')
                .sort({ 'metrics.performance': -1 });
                
            return nodes;
        } catch (error) {
            logger.error('Failed to list active nodes', { error });
            throw new Error(`Node listing failed: ${error.message}`);
        }
    }
}

module.exports = new NodeService();
