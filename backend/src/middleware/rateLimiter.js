const rateLimit = require('express-rate-limit');
const os = require('os');

const getResourceBasedLimit = () => {
  const cpuUsage = os.loadavg()[0]; // 1 minute load average
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const memUsage = (totalMem - freeMem) / totalMem;

  // Adjust limits based on resource usage
  let limit = 100; // base limit
  if (cpuUsage > 2 || memUsage > 0.8) {
    limit = Math.max(20, Math.floor(limit * 0.5));
  } else if (cpuUsage < 1 && memUsage < 0.5) {
    limit = Math.min(200, Math.floor(limit * 1.5));
  }
  return limit;
};

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts
  message: { error: 'Too many auth attempts, try again later' }
});

const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: (req) => getResourceBasedLimit(),
  message: { error: 'Too many requests, try again later' }
});

module.exports = { authLimiter, apiLimiter };
