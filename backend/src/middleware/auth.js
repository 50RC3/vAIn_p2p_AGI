const jwt = require('jsonwebtoken');
const { ethers } = require('ethers');

const verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

const verifySignature = (message, signature, address) => {
  try {
    const recoveredAddress = ethers.utils.verifyMessage(message, signature);
    return recoveredAddress.toLowerCase() === address.toLowerCase();
  } catch {
    return false;
  }
};

/**
 * Verify JWT token for Socket.io connections
 * @param {Object} socket Socket.io socket
 * @param {Function} next Next middleware function
 */
function verifySocketToken(socket, next) {
    try {
        const token = socket.handshake.auth.token || socket.handshake.headers.authorization;
        
        // For development/testing, allow connections without authentication
        if (process.env.NODE_ENV === 'development' && !token) {
            return next();
        }
        
        // Skip authentication for health check socket events
        if (socket.handshake.query && socket.handshake.query.type === 'health') {
            return next();
        }
        
        if (!token) {
            return next(new Error('Authentication token required'));
        }
        
        // Verify token - assuming the verifyToken function exists and can be adapted
        // This would be similar to your existing JWT verification logic
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        socket.user = decoded;
        next();
    } catch (err) {
        next(new Error('Invalid authentication token'));
    }
}

module.exports = { verifyToken, verifySignature, verifySocketToken };
