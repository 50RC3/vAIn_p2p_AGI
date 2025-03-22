// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorSettings.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract VAINGovernor is Governor, GovernorSettings, GovernorVotes, Pausable, Ownable {
    // Tier multipliers for vote weight
    mapping(uint256 => uint256) public tierMultipliers;
    
    // Configurable thresholds
    uint256 public minProposalThreshold;
    uint256 public maxProposalThreshold; 
    uint256 public emergencyThreshold;
    uint256 public lastConfigUpdate;
    uint256 public configCooldown;

    // Events
    event TierMultiplierUpdated(uint256 tier, uint256 multiplier);
    event ThresholdsUpdated(uint256 minThreshold, uint256 maxThreshold, uint256 emergencyThreshold);
    event EmergencyActionTriggered(address indexed triggeredBy, string reason);
    event ConfigurationUpdated(string param, uint256 value);

    // Custom errors
    error InvalidTierMultiplier();
    error InvalidThresholdValue();
    error ConfigurationCooldownActive();
    error InvalidProposalParameters();

    constructor(IVotes _token)
        Governor("VAIN Governor")
        GovernorSettings(1, 45818, 0)
        GovernorVotes(_token)
    {
        // Initialize tier multipliers (1x, 2x, 3x for tiers 1,2,3)
        tierMultipliers[1] = 100; // Base multiplier (100 = 1x)
        tierMultipliers[2] = 200; // 2x voting power
        tierMultipliers[3] = 300; // 3x voting power

        minProposalThreshold = 1000e18;  // 1000 tokens
        maxProposalThreshold = 10000e18; // 10000 tokens
        emergencyThreshold = 8000e18;    // 8000 tokens
        configCooldown = 7 days;
    }

    // Override voting weight to apply tier multipliers
    function _getVotes(
        address account,
        uint256 timepoint,
        bytes memory params
    ) internal view override returns (uint256) {
        uint256 votes = super._getVotes(account, timepoint, params);
        uint256 tier = _getUserTier(account);
        return (votes * tierMultipliers[tier]) / 100;
    }

    function updateTierMultiplier(uint256 tier, uint256 multiplier) external onlyOwner {
        if (multiplier < 100 || multiplier > 500) revert InvalidTierMultiplier();
        if (tier < 1 || tier > 3) revert InvalidTierMultiplier();
        
        tierMultipliers[tier] = multiplier;
        emit TierMultiplierUpdated(tier, multiplier);
    }

    function updateThresholds(
        uint256 _minThreshold,
        uint256 _maxThreshold,
        uint256 _emergencyThreshold
    ) external onlyOwner {
        if (block.timestamp < lastConfigUpdate + configCooldown) 
            revert ConfigurationCooldownActive();
        if (_minThreshold >= _maxThreshold || _emergencyThreshold > _maxThreshold)
            revert InvalidThresholdValue();

        minProposalThreshold = _minThreshold;
        maxProposalThreshold = _maxThreshold;
        emergencyThreshold = _emergencyThreshold;

        lastConfigUpdate = block.timestamp;
        emit ThresholdsUpdated(_minThreshold, _maxThreshold, _emergencyThreshold);
    }

    function _getUserTier(address user) internal view returns (uint256) {
        // Get user's staked amount and activity to determine tier
        // This would integrate with your staking/reputation system
        // Placeholder implementation
        uint256 stakedAmount = token().balanceOf(user);
        if (stakedAmount >= 5000e18) return 3;
        if (stakedAmount >= 1000e18) return 2;
        return 1;
    }

    // Override proposal validation with additional checks
    function propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) public override returns (uint256) {
        if (
            targets.length == 0 ||
            targets.length != values.length ||
            targets.length != calldatas.length
        ) revert InvalidProposalParameters();

        return super.propose(targets, values, calldatas, description);
    }

    function pauseGovernance() external onlyOwner {
        _pause();
    }

    function unpauseGovernance() external onlyOwner {
        _unpause();
    }

    // Override voting functions to check for paused state
    function castVote(uint256 proposalId, uint8 support)
        public
        override
        whenNotPaused
        returns (uint256)
    {
        return super.castVote(proposalId, support);
    }

    function quorum(uint256) public pure override returns (uint256) {
        return 4000e18;
    }
}
