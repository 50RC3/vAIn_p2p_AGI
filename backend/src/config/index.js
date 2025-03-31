const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');
const logger = require('../utils/logger');

// Load environment variables from .env file if exists
const envFile = process.env.NODE_ENV === 'production' ? '.env.production' : '.env';
const envPath = path.resolve(process.cwd(), envFile);

if (fs.existsSync(envPath)) {
    logger.info(`Loading environment from ${envFile}`);
    dotenv.config({ path: envPath });
} else {
    logger.warn(`Environment file ${envFile} not found, using process.env only`);
    dotenv.config(); // Try to load default .env
}

// Configuration values with defaults
const config = {
    env: process.env.NODE_ENV || 'development',
    server: {
        port: parseInt(process.env.PORT) || 8000,
        host: process.env.HOST || 'localhost',
        allowedOrigins: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        trustProxy: process.env.TRUST_PROXY === 'true',
        bodyLimit: process.env.BODY_LIMIT || '1mb'
    },
    auth: {
        jwtSecret: process.env.JWT_SECRET || 'defaultsecret', // WARN: Change in production
        jwtExpiration: process.env.JWT_EXPIRATION || '24h'
    },
    metrics: {
        interval: parseInt(process.env.METRICS_INTERVAL) || 5000, 
        retention: {
            alerts: parseInt(process.env.ALERTS_RETENTION) || 100,
            connections: parseInt(process.env.CONNECTIONS_RETENTION) || 1000
        },
        thresholds: {
            cpu: {
                warning: parseInt(process.env.CPU_WARNING_THRESHOLD) || 85,
                critical: parseInt(process.env.CPU_CRITICAL_THRESHOLD) || 95
            },
            memory: {
                warning: parseInt(process.env.MEMORY_WARNING_THRESHOLD) || 85,
                critical: parseInt(process.env.MEMORY_CRITICAL_THRESHOLD) || 95
            },
            load: {
                warning: parseFloat(process.env.LOAD_WARNING_THRESHOLD) || 2,
                critical: parseFloat(process.env.LOAD_CRITICAL_THRESHOLD) || 4
            }
        }
    },
    ipfs: {
        host: process.env.IPFS_HOST || 'localhost',
        port: process.env.IPFS_PORT || '5001',
        protocol: process.env.IPFS_PROTOCOL || 'http',
        maxRetries: parseInt(process.env.IPFS_MAX_RETRIES) || 3,
        retryDelay: parseInt(process.env.IPFS_RETRY_DELAY) || 5000,
        enableHealthCheck: process.env.IPFS_ENABLE_HEALTH !== 'false',
        healthCheckFrequency: parseInt(process.env.IPFS_HEALTH_FREQ) || 60000, // Every 60 seconds
        maxHealthHistory: parseInt(process.env.IPFS_HEALTH_HISTORY) || 100,
        features: {
            advancedPinning: process.env.IPFS_FEATURE_ADVANCED_PINNING === 'true',
            multiAddressSupport: process.env.IPFS_FEATURE_MULTI_ADDRESS === 'true',
            pubsubEnabled: process.env.IPFS_FEATURE_PUBSUB === 'true',
            gatewayProxy: process.env.IPFS_FEATURE_GATEWAY_PROXY === 'true',
        },
        pinOptions: {
            defaultReplication: parseInt(process.env.IPFS_DEFAULT_REPLICATION) || 2,
            pinningStrategy: process.env.IPFS_PINNING_STRATEGY || 'direct', // direct, service, or hybrid
            remotePinningService: process.env.IPFS_REMOTE_PINNING_SERVICE || '',
            remotePinningKey: process.env.IPFS_REMOTE_PINNING_KEY || '',
        }
    },
    rateLimiter: {
        defaultWindowMs: 60 * 1000, // 1 minute
        defaultMax: 100,
        auth: {
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 5
        }
    }
};

// Warn if JWT_SECRET is default in production
if (config.env === 'production' && config.auth.jwtSecret === 'defaultsecret') {
    logger.warn('WARNING: Using default JWT secret in production mode! Set JWT_SECRET env var.');
}

module.exports = config;
