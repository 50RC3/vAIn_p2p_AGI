// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract ReputationContract is Ownable, Pausable, ReentrancyGuard {
    mapping(address => uint256) public reputation;
    mapping(address => uint256) public lastUpdateBlock;
    uint256 public constant MIN_REPUTATION = 0;
    uint256 public constant MAX_REPUTATION = 1000;
    
    uint256 public constant DECAY_PERIOD = 30 days;
    uint256 public constant DECAY_RATE = 50; // 5% decay per period
    uint256 public constant SUSPICIOUS_UPDATE_THRESHOLD = 3;
    
    mapping(address => uint256[]) public reputationHistory;
    mapping(address => mapping(address => uint256)) public validationsPerPair;
    
    uint256 public minUpdateInterval;
    mapping(address => bool) public authorizedValidators;
    uint256 public maxBatchSize = 100;
    
    event ReputationUpdated(address indexed user, uint256 newReputation);
    event ReputationPenalty(address indexed user, uint256 penalty);
    event ValidatorAuthorized(address indexed validator, bool status);
    event ConfigurationUpdated(string parameter, uint256 value);
    event BatchReputationUpdated(uint256 count);
    event EmergencyShutdown(address indexed trigger, string reason);
    
    modifier onlyValidator() {
        require(authorizedValidators[msg.sender], "Not authorized validator");
        _;
    }
    
    modifier validReputation(uint256 score) {
        require(score <= MAX_REPUTATION, "Score exceeds maximum");
        require(score >= MIN_REPUTATION, "Score below minimum");
        _;
    }

    function safeUpdateReputation(
        address user,
        uint256 score,
        string memory reason
    ) external nonReentrant whenNotPaused onlyValidator validReputation(score) {
        require(user != address(0), "Invalid user address");
        require(block.timestamp >= lastUpdateBlock[user] + minUpdateInterval, "Update too frequent");
        
        // Apply time decay
        uint256 currentRep = _calculateDecayedReputation(user);
        
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
    
    function batchUpdateReputation(
        address[] calldata users,
        uint256[] calldata scores
    ) external nonReentrant whenNotPaused onlyValidator {
        require(users.length == scores.length, "Length mismatch");
        require(users.length <= maxBatchSize, "Batch too large");
        
        for (uint i = 0; i < users.length; i++) {
            if (_isValidUpdate(users[i], scores[i])) {
                reputation[users[i]] = scores[i];
                lastUpdateBlock[users[i]] = block.timestamp;
                emit ReputationUpdated(users[i], scores[i]);
            }
        }
        
        emit BatchReputationUpdated(users.length);
    }
    
    function _calculateDecayedReputation(address user) internal view returns (uint256) {
        uint256 timePassed = block.timestamp - lastUpdateBlock[user];
        uint256 decayPeriods = timePassed / DECAY_PERIOD;
        uint256 currentRep = reputation[user];
        
        for (uint i = 0; i < decayPeriods; i++) {
            currentRep = currentRep * (1000 - DECAY_RATE) / 1000;
        }
        
        return currentRep;
    }
    
    function _isValidUpdate(address user, uint256 score) internal view returns (bool) {
        return user != address(0) && 
               score <= MAX_REPUTATION &&
               score >= MIN_REPUTATION &&
               block.timestamp >= lastUpdateBlock[user] + minUpdateInterval;
    }
    
    // Admin functions
    function setValidator(address validator, bool status) external onlyOwner {
        authorizedValidators[validator] = status;
        emit ValidatorAuthorized(validator, status);
    }
    
    function setMinUpdateInterval(uint256 interval) external onlyOwner {
        minUpdateInterval = interval;
        emit ConfigurationUpdated("minUpdateInterval", interval);
    }
    
    function setMaxBatchSize(uint256 size) external onlyOwner {
        require(size > 0 && size <= 200, "Invalid batch size");
        maxBatchSize = size;
        emit ConfigurationUpdated("maxBatchSize", size);
    }
    
    // Emergency functions
    function emergencyPause(string calldata reason) external onlyOwner {
        _pause();
        emit EmergencyShutdown(msg.sender, reason);
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    // View functions for frontend
    function getReputationWithDecay(address user) external view returns (uint256) {
        return _calculateDecayedReputation(user);
    }
    
    function getReputationHistory(address user) external view returns (uint256[] memory) {
        return reputationHistory[user];
    }
    
    function getUserValidations(address validator, address user) external view returns (uint256) {
        return validationsPerPair[validator][user];
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
