const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const logger = require('./src/utils/logger');
const { apiLimiter } = require('./src/middleware/rateLimiter');
const { verifyToken } = require('./src/middleware/auth');
const { metricsMiddleware } = require('./src/middleware/metrics');

// Initialize express
const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || 'http://localhost:3000',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(metricsMiddleware);

// Global rate limiting
app.use(apiLimiter);

// Request parsing
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(morgan('combined', { stream: logger.stream }));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: Date.now() });
});

// Metrics endpoint
app.get('/metrics', verifyToken, (req, res) => {
    const metrics = require('./src/middleware/metrics').getMetrics();
    res.json(metrics);
});

// Protected routes
app.use('/api', verifyToken);

// API Routes
app.use('/api/auth', require('./src/api/auth'));
app.use('/api/staking', require('./src/api/staking'));

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
const server = app.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
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

module.exports = app;
