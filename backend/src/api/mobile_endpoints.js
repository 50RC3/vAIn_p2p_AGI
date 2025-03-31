const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');
const { pythonService } = require('../services/python');
const ipfs = require('../services/ipfs');

/**
 * Android chatbot endpoint that processes messages and returns responses
 */
router.post('/chat/android', async (req, res) => {
    try {
        const { message, user_id } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }
        
        // Process message through Python backend with AndroidChatInterface
        const response = await pythonService.call('android_chat_interface.process_message', {
            message,
            user_id: user_id || 'android_user'
        });
        
        // Log the interaction for metrics
        logger.debug('Android chat message processed', { user_id, responseTime: response.processing_time });
        
        // Store interaction in IPFS for later analysis (with privacy consideration)
        try {
            await ipfs.store({
                type: 'android_chat',
                timestamp: Date.now(),
                message_length: message.length,
                response_length: response.text.length,
                confidence: response.confidence,
                processing_time: response.processing_time
            }, { 
                type: 'metrics',
                platform: 'android'
            });
        } catch (ipfsError) {
            // Non-fatal error, just log it
            logger.warn('Failed to store chat metrics in IPFS', { error: ipfsError.message });
        }
        
        return res.json(response);
        
    } catch (error) {
        logger.error('Error processing Android chat message', { error });
        return res.status(500).json({ 
            error: 'Failed to process message',
            message: error.message
        });
    }
});

/**
 * Endpoint to sync offline messages
 */
router.post('/chat/android/sync', async (req, res) => {
    try {
        const { user_id, offline_messages } = req.body;
        
        if (!offline_messages || !Array.isArray(offline_messages)) {
            return res.status(400).json({ error: 'Valid offline_messages array required' });
        }
        
        // Process offline messages in batch
        const results = await pythonService.call('android_chat_interface.sync_offline_messages', {
            user_id: user_id || 'android_user',
            messages: offline_messages
        });
        
        return res.json({ results });
        
    } catch (error) {
        logger.error('Error syncing offline messages', { error });
        return res.status(500).json({ error: 'Failed to sync offline messages' });
    }
});

/**
 * Endpoint to check connection and chatbot status
 */
router.get('/chat/android/status', async (req, res) => {
    try {
        const ipfsStatus = await ipfs.checkHealth();
        
        return res.json({
            status: 'online',
            timestamp: Date.now(),
            ipfs: {
                connected: ipfsStatus.connected,
                responseTime: ipfsStatus.responseTime
            },
            chatbot: {
                ready: true,
                version: process.env.APP_VERSION || '0.1.0'
            }
        });
    } catch (error) {
        logger.error('Error getting system status', { error });
        return res.status(500).json({ error: 'Failed to get system status' });
    }
});

module.exports = router;
