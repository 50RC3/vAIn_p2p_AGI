const express = require('express');
const router = express.Router();
const { ethers } = require('ethers');
const { calculateRewards } = require('../utils/reward-calculator');

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
router.get('/pending', async (req, res) => {
    try {
        const { address } = req.user;
        
        if (!ethers.utils.isAddress(address)) {
            return res.status(400).json({ 
                error: "Invalid ethereum address" 
            });
        }

        const rewards = await calculateRewards(address);
        res.json({ rewards: rewards.toString() });
    } catch (error) {
        res.status(500).json({ 
            error: "Failed to calculate rewards",
            details: error.message 
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
router.post('/claim', async (req, res) => {
    try {
        const { address } = req.user;

        if (!ethers.utils.isAddress(address)) {
            return res.status(400).json({ 
                error: "Invalid ethereum address" 
            });
        }

        // Implementation for claiming rewards
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ 
            error: "Failed to claim rewards",
            details: error.message 
        });
    }
});

module.exports = router;
