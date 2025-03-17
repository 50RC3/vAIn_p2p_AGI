// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/Ownable.sol";

contract ReputationContract is Ownable {
    mapping(address => uint256) public reputation;
    mapping(address => uint256) public lastUpdateBlock;
    uint256 public constant MIN_REPUTATION = 0;
    uint256 public constant MAX_REPUTATION = 1000;
    
    uint256 public constant DECAY_PERIOD = 30 days;
    uint256 public constant DECAY_RATE = 50; // 5% decay per period
    uint256 public constant SUSPICIOUS_UPDATE_THRESHOLD = 3;
    
    mapping(address => uint256[]) public reputationHistory;
    mapping(address => mapping(address => uint256)) public validationsPerPair;
    
    event ReputationUpdated(address indexed user, uint256 newReputation);
    event ReputationPenalty(address indexed user, uint256 penalty);
    
    function updateReputation(address user, uint256 score) external onlyOwner {
        require(score <= MAX_REPUTATION, "Score exceeds maximum");
        
        // Apply time decay
        uint256 timePassed = block.timestamp - lastUpdateBlock[user];
        uint256 decayPeriods = timePassed / DECAY_PERIOD;
        uint256 currentRep = reputation[user];
        
        for (uint i = 0; i < decayPeriods; i++) {
            currentRep = currentRep * (1000 - DECAY_RATE) / 1000;
        }
        
        // Check for suspicious patterns
        if (detectCollusion(msg.sender, user)) {
            emit ReputationPenalty(user, 100);
            reputation[user] = currentRep > 100 ? currentRep - 100 : 0;
            return;
        }
        
        reputation[user] = score;
        reputationHistory[user].push(score);
        lastUpdateBlock[user] = block.timestamp;
        validationsPerPair[msg.sender][user]++;
        
        emit ReputationUpdated(user, score);
    }
    
    function detectCollusion(address validator, address user) internal view returns (bool) {
        if (validationsPerPair[validator][user] > SUSPICIOUS_UPDATE_THRESHOLD) {
            return true;
        }
        return false;
    }
    
    function applyPenalty(address user, uint256 penalty) external onlyOwner {
        uint256 currentRep = reputation[user];
        uint256 newRep = currentRep > penalty ? currentRep - penalty : MIN_REPUTATION;
        reputation[user] = newRep;
        emit ReputationPenalty(user, penalty);
    }
}
