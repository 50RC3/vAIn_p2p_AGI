const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const http = require('http');
const socketIo = require('socket.io');
const logger = require('./src/utils/logger');
const { apiLimiter, createEndpointLimiter } = require('./src/middleware/rateLimiter');
const { verifyToken, verifySocketToken } = require('./src/middleware/auth');
const { metricsMiddleware } = require('./src/middleware/metrics');
const { startMetricsMonitoring } = require('./src/services/monitoringService');

// Initialize express
const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        methods: ['GET', 'POST'],
        credentials: false
    }
});

// Socket.io authentication and setup
io.use(verifySocketToken);
require('./src/socket')(io); // Setup socket handlers

// Security middleware - fine-tuned helmet configuration
app.use(helmet({
    // Keep essential security headers
    contentSecurityPolicy: true,  // Helps prevent XSS attacks
    crossOriginEmbedderPolicy: true,
    crossOriginOpenerPolicy: true,
    crossOriginResourcePolicy: true,
    
    // Disable unnecessary headers
    dnsPrefetchControl: false,  // Not needed for most APIs
    expectCt: false,  // Certificate Transparency being phased out
    originAgentCluster: false,  // Experimental feature
    
    // Keep important headers
    referrerPolicy: true,
    hsts: true,  // HTTP Strict Transport Security
    noSniff: true,  // X-Content-Type-Options
    frameguard: true,  // X-Frame-Options to prevent clickjacking
    xssFilter: true,  // Basic XSS protection
}));

// Restricted CORS configuration for enhanced security
app.use(cors({
    // Only allow specific origins instead of wildcard
    origin: function(origin, callback) {
        const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
        // Allow requests with no origin (like mobile apps or curl requests)
        if (!origin || allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('CORS policy violation'));
        }
    },
    // Only allow necessary HTTP methods
    methods: ['GET', 'POST'],
    // Explicitly list allowed headers
    allowedHeaders: ['Content-Type', 'Authorization'],
    // Don't allow cookies by default (change to true if needed)
    credentials: false,
    // Limit which headers can be read by frontend
    exposedHeaders: ['X-Total-Count'],
    // Cache preflight requests for 1 hour (in seconds)
    maxAge: 3600,
    // Don't allow browser to send credentials like cookies or auth headers
    optionsSuccessStatus: 204 // Standard for preflight responses
}));

app.use(metricsMiddleware);

// Global rate limiting
app.use(apiLimiter);

// Request parsing
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(morgan('combined', { stream: logger.stream }));

// Create endpoint-specific rate limiters with different thresholds
const healthCheckLimiter = createEndpointLimiter(100, 60); // 100 requests per minute
const metricsLimiter = createEndpointLimiter(30, 60);     // 30 requests per minute
const authLimiter = createEndpointLimiter(20, 60);        // 20 requests per minute
const stakingLimiter = createEndpointLimiter(10, 60);     // 10 requests per minute
const nodesLimiter = createEndpointLimiter(50, 60);       // 50 requests per minute
const ipfsLimiter = createEndpointLimiter(40, 60);        // 40 requests per minute

// Health check endpoint with specific rate limit
app.get('/health', healthCheckLimiter, (req, res) => {
    res.json({ status: 'ok', timestamp: Date.now() });
});

// Metrics endpoint with specific rate limit
app.get('/metrics', metricsLimiter, verifyToken, (req, res) => {
    const metrics = require('./src/middleware/metrics').getMetrics();
    res.json(metrics);
});

// Protected routes
app.use('/api', verifyToken);

// API Routes with endpoint-specific rate limits
app.use('/api/auth', authLimiter, require('./src/api/auth'));
app.use('/api/staking', stakingLimiter, require('./src/api/staking'));
app.use('/api/nodes', nodesLimiter, require('./src/api/nodes'));
app.use('/api/ipfs', ipfsLimiter, require('./src/api/ipfs'));

// Error handling middleware
app.use((err, req, res, next) => {
    logger.error(`Unhandled error: ${err.stack}`);
    res.status(err.status || 500).json({
        error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message,
        timestamp: Date.now()
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        error: 'Not found',
        timestamp: Date.now()
    });
});

// Start server
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
    
    // Start real-time monitoring service
    startMetricsMonitoring(io);
    logger.info('Real-time monitoring service started');
});

// Graceful shutdown
process.on('SIGTERM', () => {
    logger.info('SIGTERM signal received. Shutting down gracefully...');
    server.close(() => {
        logger.info('Server closed');
        process.exit(0);
    });

    // Force close after 10s
    setTimeout(() => {
        logger.error('Could not close server gracefully, forcefully shutting down');
        process.exit(1);
    }, 10000);
});

process.on('uncaughtException', (err) => {
    logger.error('Uncaught exception:', err);
    process.exit(1);
});

process.on('unhandledRejection', (err) => {
    logger.error('Unhandled rejection:', err);
    process.exit(1);
});

module.exports = { app, server, io };
