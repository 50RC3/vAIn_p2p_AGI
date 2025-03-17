const express = require('express');
const router = express.Router();
const { ethers } = require('ethers');
const ReputationContract = require('../../contracts/reputationContract.json');

router.get('/score/:address', async (req, res) => {
    try {
        const { address } = req.params;
        const contract = new ethers.Contract(
            process.env.REPUTATION_CONTRACT_ADDRESS,
            ReputationContract.abi,
            provider
        );
        const score = await contract.reputation(address);
        res.json({ score: score.toString() });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;
