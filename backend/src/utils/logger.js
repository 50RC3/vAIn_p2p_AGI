const winston = require('winston');
const path = require('path');

// Create logs directory if it doesn't exist
const logDir = path.join(__dirname, '../../../logs');
require('fs').mkdirSync(logDir, { recursive: true });

const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ 
            filename: path.join(logDir, 'error.log'), 
            level: 'error' 
        }),
        new winston.transports.File({ 
            filename: path.join(logDir, 'combined.log')
        })
    ]
});

// Add console logging in non-production environments
if (process.env.NODE_ENV !== 'production') {
    logger.add(new winston.transports.Console({
        format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
        )
    }));
}

// Add stream for Morgan middleware
logger.stream = {
    write: (message) => logger.info(message.trim())
};

module.exports = logger;
