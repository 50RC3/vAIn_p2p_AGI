const os = require('os');
const logger = require('../utils/logger');
const config = require('../config');
const ipfsService = require('./ipfs');

// Use metrics configuration from centralized config
const METRICS_INTERVAL = config.metrics.interval;
const ALERT_THRESHOLDS = {
    CPU_WARNING: config.metrics.thresholds.cpu.warning,
    CPU_CRITICAL: config.metrics.thresholds.cpu.critical,
    MEMORY_WARNING: config.metrics.thresholds.memory.warning,
    MEMORY_CRITICAL: config.metrics.thresholds.memory.critical,
    LOAD_WARNING: config.metrics.thresholds.load.warning,
    LOAD_CRITICAL: config.metrics.thresholds.load.critical
};

// Track metrics and connections
let metricsInterval;
let activeConnections = 0;
let connectionHistory = [];
let alertsHistory = [];

/**
 * Starts the real-time monitoring service
 * @param {Object} io Socket.io instance
 */
function startMetricsMonitoring(io) {
    if (metricsInterval) {
        clearInterval(metricsInterval);
    }

    // Track socket connections
    io.on('connection', (socket) => {
        activeConnections++;
        
        // Keep connection history for the last 24 hours
        connectionHistory.push({
            time: Date.now(),
            count: activeConnections
        });
        
        // Trim history to last 24 hours
        const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
        connectionHistory = connectionHistory.filter(entry => entry.time > oneDayAgo);

        // Send current metrics upon connection
        const metrics = collectMetrics();
        socket.emit('metrics:update', metrics);

        socket.on('disconnect', () => {
            activeConnections--;
        });
    });

    // Set up IPFS monitoring if available
    setupIPFSMonitoring(io);

    // Broadcast metrics at regular intervals
    metricsInterval = setInterval(() => {
        try {
            const metrics = collectMetrics();
            io.emit('metrics:update', metrics);
            
            // Check for alerts
            checkAlerts(metrics, io);
            
        } catch (err) {
            logger.error('Error in metrics collection:', err);
        }
    }, METRICS_INTERVAL);
}

/**
 * Set up monitoring for IPFS connections
 * @param {Object} io Socket.io instance
 */
function setupIPFSMonitoring(io) {
    // Listen for IPFS connection state changes
    ipfsService.on('connectionStateChanged', async (event) => {
        // Create an alert for connection state changes
        const alert = {
            type: event.currentState ? 'info' : 'critical',
            source: 'ipfs',
            message: event.currentState 
                ? 'IPFS connection restored' 
                : 'IPFS connection lost',
            timestamp: event.timestamp,
            details: {
                previousState: event.previousState,
                currentState: event.currentState
            }
        };
        
        // Add to alerts history
        alertsHistory.push(alert);
        
        // Keep alert history to configured size
        if (alertsHistory.length > config.metrics.retention.alerts) {
            alertsHistory = alertsHistory.slice(-config.metrics.retention.alerts);
        }
        
        // Broadcast the alert
        io.emit('metrics:alert', [alert]);
        
        // Log the alert
        if (alert.type === 'critical') {
            logger.warn(`CRITICAL ALERT: ${alert.message}`);
        } else {
            logger.info(`INFO ALERT: ${alert.message}`);
        }
        
        // Also send an updated metrics collection with the IPFS status
        const metrics = collectMetrics();
        io.emit('metrics:update', metrics);
    });
}

/**
 * Collect system metrics
 * @returns {Object} System metrics
 */
function collectMetrics() {
    const loadAvg = os.loadavg();
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const memoryUsed = ((totalMem - freeMem) / totalMem) * 100;
    const cpuUsage = getCpuUsage();
    
    // Get IPFS health information if available
    let ipfsHealth = { status: 'unknown' };
    try {
        ipfsHealth = ipfsService.getHealthSummary(10); // Last 10 minutes
    } catch (err) {
        logger.debug('Error collecting IPFS health metrics:', err);
    }
    
    return {
        timestamp: Date.now(),
        system: {
            cpuUsage: cpuUsage,
            memoryUsage: memoryUsed.toFixed(2),
            totalMemory: formatBytes(totalMem),
            freeMemory: formatBytes(freeMem),
            loadAverage: loadAvg,
            uptime: os.uptime(),
            platform: os.platform(),
            hostname: os.hostname()
        },
        network: {
            activeConnections: activeConnections,
            connectionHistory: connectionHistory.slice(-20) // Last 20 entries
        },
        alerts: alertsHistory.slice(-5), // Last 5 alerts
        services: {
            ipfs: ipfsHealth
        }
    };
}

/**
 * Get approximate CPU usage
 * This is a simple approximation - for more accurate metrics in production,
 * consider using a more sophisticated CPU monitoring library
 */
function getCpuUsage() {
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;
    
    for (const cpu of cpus) {
        for (const type in cpu.times) {
            totalTick += cpu.times[type];
        }
        totalIdle += cpu.times.idle;
    }
    
    // Return approximation based on idle time (100% - idle%)
    return Math.round(100 - (totalIdle / totalTick) * 100);
}

/**
 * Format bytes to human-readable format
 * @param {Number} bytes The number of bytes
 * @returns {String} Formatted string
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Check for system alerts based on metrics
 * @param {Object} metrics The collected system metrics
 * @param {Object} io Socket.io instance for broadcasting alerts
 */
function checkAlerts(metrics, io) {
    const alerts = [];
    const { cpuUsage, memoryUsage } = metrics.system;
    const loadAvg = metrics.system.loadAverage[0];
    
    // CPU alerts
    if (cpuUsage > ALERT_THRESHOLDS.CPU_CRITICAL) {
        const alert = {
            type: 'critical',
            source: 'cpu',
            message: `Critical CPU usage: ${cpuUsage}%`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    } else if (cpuUsage > ALERT_THRESHOLDS.CPU_WARNING) {
        const alert = {
            type: 'warning',
            source: 'cpu',
            message: `High CPU usage: ${cpuUsage}%`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    }
    
    // Memory alerts
    if (memoryUsage > ALERT_THRESHOLDS.MEMORY_CRITICAL) {
        const alert = {
            type: 'critical',
            source: 'memory',
            message: `Critical memory usage: ${memoryUsage}%`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    } else if (memoryUsage > ALERT_THRESHOLDS.MEMORY_WARNING) {
        const alert = {
            type: 'warning',
            source: 'memory',
            message: `High memory usage: ${memoryUsage}%`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    }
    
    // System load alerts
    if (loadAvg > ALERT_THRESHOLDS.LOAD_CRITICAL) {
        const alert = {
            type: 'critical',
            source: 'load',
            message: `Critical system load: ${loadAvg.toFixed(2)}`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    } else if (loadAvg > ALERT_THRESHOLDS.LOAD_WARNING) {
        const alert = {
            type: 'warning',
            source: 'load',
            message: `High system load: ${loadAvg.toFixed(2)}`,
            timestamp: Date.now()
        };
        alerts.push(alert);
        alertsHistory.push(alert);
    }
    
    // Keep alert history to configured size
    if (alertsHistory.length > config.metrics.retention.alerts) {
        alertsHistory = alertsHistory.slice(-config.metrics.retention.alerts);
    }
    
    // Broadcast alerts if any
    if (alerts.length > 0) {
        io.emit('metrics:alert', alerts);
        
        // Log critical alerts
        alerts.forEach(alert => {
            if (alert.type === 'critical') {
                logger.warn(`CRITICAL ALERT: ${alert.message}`);
            }
        });
    }
}

function stopMetricsMonitoring() {
    if (metricsInterval) {
        clearInterval(metricsInterval);
        metricsInterval = null;
    }
}

module.exports = {
    startMetricsMonitoring,
    stopMetricsMonitoring,
    collectMetrics
};
