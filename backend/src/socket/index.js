const logger = require('../utils/logger');
const { createEndpointLimiter } = require('../middleware/rateLimiter');

/**
 * Configure Socket.io event handlers
 * @param {Object} io Socket.io server instance
 */
module.exports = function(io) {
    // Socket rate limiting (using memory adapter by default)
    const socketRateLimiter = createEndpointLimiter(30, 60);
    
    // Socket middleware
    io.use((socket, next) => {
        // Apply rate limiting to socket connections
        socketRateLimiter(socket.request, {}, next);
    });

    // Handle new connections
    io.on('connection', (socket) => {
        logger.info(`Socket connected: ${socket.id}`);
        
        // Handle subscription to metrics updates
        socket.on('subscribe:metrics', (options = {}) => {
            const interval = Math.max(1000, Math.min(30000, options.interval || 5000));
            socket.join('metrics-subscribers');
            logger.debug(`Socket ${socket.id} subscribed to metrics with interval ${interval}ms`);
            
            // Send confirmation
            socket.emit('subscription:confirmed', {
                channel: 'metrics',
                interval: interval
            });
        });
        
        // Handle subscription to alerts
        socket.on('subscribe:alerts', () => {
            socket.join('alerts-subscribers');
            logger.debug(`Socket ${socket.id} subscribed to alerts`);
            
            // Send confirmation
            socket.emit('subscription:confirmed', {
                channel: 'alerts'
            });
        });
        
        // Handle unsubscription from metrics
        socket.on('unsubscribe:metrics', () => {
            socket.leave('metrics-subscribers');
            logger.debug(`Socket ${socket.id} unsubscribed from metrics`);
        });
        
        // Handle unsubscription from alerts
        socket.on('unsubscribe:alerts', () => {
            socket.leave('alerts-subscribers');
            logger.debug(`Socket ${socket.id} unsubscribed from alerts`);
        });
        
        // Handle disconnect
        socket.on('disconnect', () => {
            logger.debug(`Socket disconnected: ${socket.id}`);
        });
    });
    
    return io;
};
