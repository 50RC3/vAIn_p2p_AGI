const rateLimit = require('express-rate-limit');
const os = require('os');
const config = require('../config');

const getResourceBasedLimit = () => {
  const cpuUsage = os.loadavg()[0]; // 1 minute load average
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const memUsage = (totalMem - freeMem) / totalMem;

  // Adjust limits based on resource usage
  let limit = config.rateLimiter.defaultMax; // base limit from config
  if (cpuUsage > 2 || memUsage > 0.8) {
    limit = Math.max(20, Math.floor(limit * 0.5));
  } else if (cpuUsage < 1 && memUsage < 0.5) {
    limit = Math.min(200, Math.floor(limit * 1.5));
  }
  return limit;
};

const authLimiter = rateLimit({
  windowMs: config.rateLimiter.auth.windowMs,
  max: config.rateLimiter.auth.max,
  message: { error: 'Too many auth attempts, try again later' }
});

const apiLimiter = rateLimit({
  windowMs: config.rateLimiter.defaultWindowMs,
  max: (req) => getResourceBasedLimit(),
  message: { error: 'Too many requests, try again later' }
});

// Function to create endpoint-specific rate limiters
const createEndpointLimiter = (maxRequests, windowSeconds = 60) => {
  return rateLimit({
    windowMs: windowSeconds * 1000,
    max: (req) => {
      const baseLimit = typeof maxRequests === 'function' ? maxRequests(req) : maxRequests;
      
      // Apply resource-based adjustment
      const cpuUsage = os.loadavg()[0];
      const memUsage = (os.totalmem() - os.freemem()) / os.totalmem();
      
      // Reduce limits during high resource usage
      if (cpuUsage > 2.5 || memUsage > 0.85) {
        return Math.max(5, Math.floor(baseLimit * 0.4)); // Drastic reduction during high load
      } else if (cpuUsage > 1.5 || memUsage > 0.7) {
        return Math.max(10, Math.floor(baseLimit * 0.7)); // Moderate reduction
      }
      
      return baseLimit;
    },
    message: { error: 'Rate limit exceeded for this endpoint, please try again later' },
    standardHeaders: true, // Return rate limit info in the headers
    legacyHeaders: false   // Disable legacy X-RateLimit headers
  });
};

module.exports = { authLimiter, apiLimiter, createEndpointLimiter };
