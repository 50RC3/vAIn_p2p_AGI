const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const { ethers } = require('ethers');
const { body, validationResult } = require('express-validator');

// Rate limiting
const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts per window
    message: { error: 'Too many login attempts, please try again later' }
});

// Validation middleware
const validateLogin = [
    body('address').isEthereumAddress(),
    body('signature').isString().isLength({ min: 132, max: 132 }),
];

router.post('/login', loginLimiter, validateLogin, async (req, res) => {
    try {
        // Validation check
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const { address, signature } = req.body;

        // Verify Ethereum signature
        const message = `Login to vAIn: ${new Date().toISOString().slice(0,10)}`;
        const recoveredAddress = ethers.utils.verifyMessage(message, signature);

        if (recoveredAddress.toLowerCase() !== address.toLowerCase()) {
            return res.status(401).json({ error: 'Invalid signature' });
        }

        // Create tokens
        const accessToken = jwt.sign(
            { address }, 
            process.env.JWT_SECRET, 
            { expiresIn: '1h' }
        );

        const refreshToken = jwt.sign(
            { address },
            process.env.REFRESH_SECRET,
            { expiresIn: '7d' }
        );

        res.json({
            accessToken,
            refreshToken,
            expiresIn: 3600, // 1 hour in seconds
        });

    } catch (error) {
        console.error('Auth error:', error);
        res.status(500).json({ 
            error: 'Authentication failed',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
});

router.post('/refresh', rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 10 // 10 attempts per hour
}), async (req, res) => {
    try {
        const { refreshToken } = req.body;
        
        if (!refreshToken) {
            return res.status(400).json({ error: 'Refresh token required' });
        }

        const decoded = jwt.verify(refreshToken, process.env.REFRESH_SECRET);
        const accessToken = jwt.sign(
            { address: decoded.address },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        res.json({
            accessToken,
            expiresIn: 3600
        });

    } catch (error) {
        if (error instanceof jwt.TokenExpiredError) {
            return res.status(401).json({ error: 'Refresh token expired' });
        }
        res.status(500).json({ error: 'Token refresh failed' });
    }
});

module.exports = router;
