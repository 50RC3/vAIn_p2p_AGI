const logger = require('../utils/logger');

const apiMetrics = {
  requestCount: 0,
  errorCount: 0,
  responseTimeTotal: 0,
  requestsByEndpoint: {},
  lastReset: Date.now()
};

const metricsMiddleware = (req, res, next) => {
  const startTime = Date.now();
  const endpoint = `${req.method} ${req.path}`;

  // Initialize endpoint metrics if needed
  if (!apiMetrics.requestsByEndpoint[endpoint]) {
    apiMetrics.requestsByEndpoint[endpoint] = {
      count: 0,
      errors: 0,
      totalTime: 0
    };
  }

  // Count the request
  apiMetrics.requestCount++;
  apiMetrics.requestsByEndpoint[endpoint].count++;

  // Add response listener
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    apiMetrics.responseTimeTotal += duration;
    apiMetrics.requestsByEndpoint[endpoint].totalTime += duration;

    // Track errors
    if (res.statusCode >= 400) {
      apiMetrics.errorCount++;
      apiMetrics.requestsByEndpoint[endpoint].errors++;
    }

    // Log securely
    logger.info('API Request', {
      endpoint,
      duration,
      status: res.statusCode,
      encrypted: true,
      timestamp: new Date().toISOString()
    });
  });

  next();
};

const getMetrics = () => ({
  ...apiMetrics,
  avgResponseTime: apiMetrics.responseTimeTotal / apiMetrics.requestCount || 0,
  errorRate: apiMetrics.errorCount / apiMetrics.requestCount || 0,
  uptime: Date.now() - apiMetrics.lastReset
});

module.exports = { metricsMiddleware, getMetrics };
