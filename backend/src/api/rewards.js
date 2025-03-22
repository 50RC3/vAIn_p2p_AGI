const express = require('express');
const router = express.Router();
const { ethers } = require('ethers');
const rateLimit = require('express-rate-limit');
const { calculateRewards } = require('../utils/reward-calculator');
const logger = require('../utils/logger');
const metrics = require('../utils/metrics');

// Initialize provider
const provider = new ethers.providers.JsonRpcProvider(process.env.ETH_RPC_URL);

// Rate limiting
const rateLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 20 // Maximum 20 requests per window
});

// Validation middleware
const validateRequest = (req, res, next) => {
    const { address } = req.user;
    if (!ethers.utils.isAddress(address)) {
        return res.status(400).json({ error: 'Invalid Ethereum address' });
    }
    next();
};

/**
 * @typedef {Object} RewardsResponse
 * @property {string} rewards - The pending rewards amount
 */

/**
 * Get pending rewards for a user
 * @route GET /rewards/pending
 * @param {Object} req.user - Authenticated user object
 * @returns {RewardsResponse} Pending rewards
 * @throws {400} If user address is invalid
 * @throws {500} If calculation fails
 */
router.get('/pending', rateLimiter, validateRequest, async (req, res) => {
    const startTime = Date.now();
    try {
        const { address } = req.user;
        
        // Add timeout for reward calculation
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), 5000);
        });

        const rewards = await Promise.race([
            calculateRewards(address),
            timeoutPromise
        ]);

        // Cache headers
        res.set('Cache-Control', 'private, max-age=60');
        
        metrics.recordRewardsCheck(address, Date.now() - startTime);
        
        res.json({ 
            rewards: rewards.toString(),
            timestamp: Date.now()
        });

    } catch (error) {
        logger.error(`Rewards error for ${req.user.address}: ${error}`);
        
        if (error.message === 'Request timeout') {
            return res.status(504).json({ error: 'Request timed out' });
        }
        if (error.code === 'NETWORK_ERROR') {
            return res.status(503).json({ error: 'Network error' });
        }
        
        res.status(500).json({ 
            error: 'Failed to calculate rewards',
            timestamp: Date.now()
        });
    }
});

/**
 * Claim pending rewards
 * @route POST /rewards/claim
 * @param {Object} req.user - Authenticated user object
 * @returns {Object} Success status
 * @throws {400} If user address is invalid
 * @throws {500} If claim fails
 */
router.post('/claim', rateLimiter, validateRequest, async (req, res) => {
    const startTime = Date.now();
    try {
        const { address } = req.user;
        
        // Get contract instance
        const rewardsContract = new ethers.Contract(
            process.env.REWARDS_CONTRACT_ADDRESS,
            RewardsContractABI,
            provider.getSigner(address)
        );

        const tx = await rewardsContract.claimRewards();
        await tx.wait(); // Wait for confirmation

        metrics.recordRewardsClaim(address, tx.hash, Date.now() - startTime);

        res.json({
            success: true,
            txHash: tx.hash,
            timestamp: Date.now()
        });

    } catch (error) {
        logger.error(`Claim error for ${req.user.address}: ${error}`);
        
        if (error.code === 4001) { // User rejected
            return res.status(400).json({ error: 'Transaction rejected' });
        }
        
        res.status(500).json({
            error: 'Failed to claim rewards',
            timestamp: Date.now()
        });
    }
});

module.exports = router;
