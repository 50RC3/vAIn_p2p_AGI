const express = require('express');
const router = express.Router();
const { ethers } = require('ethers');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const StakingContract = require('../../contracts/stakingContract.json');

// Rate limiting
const stakingLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 staking attempts per window
    message: { error: 'Too many staking attempts, please try again later' }
});

// Validation middleware
const validateStake = [
    body('amount').isString().matches(/^\d+$/).withMessage('Amount must be a valid number string'),
    body('address').isEthereumAddress().withMessage('Invalid Ethereum address')
];

router.post('/stake', stakingLimiter, validateStake, async (req, res) => {
    try {
        // Validation check
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const { amount, address } = req.body;
        
        // Initialize provider and contract
        const provider = new ethers.providers.JsonRpcProvider(process.env.ETH_RPC_URL);
        const contract = new ethers.Contract(
            process.env.STAKING_CONTRACT_ADDRESS,
            StakingContract.abi,
            provider.getSigner(address)
        );

        // Add timeout for contract interaction
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), 30000);
        });

        // Attempt staking transaction
        const tx = await Promise.race([
            contract.stake({ value: amount }),
            timeoutPromise
        ]);

        // Wait for confirmation with timeout
        await tx.wait(1); // Wait for 1 confirmation

        return res.json({
            success: true,
            txHash: tx.hash,
            amount,
            timestamp: Date.now()
        });

    } catch (error) {
        console.error('Staking error:', error);

        // Handle specific error cases
        if (error.message.includes('insufficient funds')) {
            return res.status(400).json({ 
                error: 'Insufficient funds for staking',
                details: error.message
            });
        }
        
        if (error.message === 'Request timeout') {
            return res.status(504).json({ 
                error: 'Transaction timeout',
                details: 'The staking request timed out' 
            });
        }

        if (error.code === 'NETWORK_ERROR') {
            return res.status(503).json({ 
                error: 'Network error',
                details: 'Unable to connect to blockchain network' 
            });
        }

        if (error.code === 4001) { // User rejected
            return res.status(400).json({ 
                error: 'Transaction rejected',
                details: 'User rejected the transaction'
            });
        }

        res.status(500).json({
            error: 'Internal server error',
            details: error.message,
            timestamp: Date.now()
        });
    }
});

module.exports = router;
