// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract StakingContract is ReentrancyGuard, Pausable, Ownable {
    IERC20 public immutable stakingToken;
    
    struct Stake {
        uint256 amount;
        uint256 startTime;
        uint256 lockPeriod;
        uint256 rewardMultiplier;
    }
    
    struct Appeal {
        uint256 slashAmount;
        uint256 timestamp;
        string reason;
        bool resolved;
        bool accepted;
    }
    
    mapping(address => Stake) public stakes;
    
    // Lock period multipliers (in days)
    mapping(uint256 => uint256) public lockPeriodMultipliers;
    
    mapping(address => Appeal[]) public appeals;
    uint256 public constant APPEAL_WINDOW = 7 days;
    uint256 public constant APPEAL_FEE = 0.1 ether;
    
    uint256 public constant MIN_STAKE = 1000 * 1e18; // 1000 VAIN
    uint256 public constant BASE_APR = 500; // 5%
    
    // Add validation constants
    uint256 public constant MAX_LOCK_PERIOD = 365 days;
    uint256 public constant MIN_LOCK_PERIOD = 30 days;
    uint256 public constant MAX_APPEALS_PER_USER = 3;
    uint256 public constant COOLDOWN_PERIOD = 1 days;
    
    // Add tracking
    mapping(address => uint256) public lastStakeTime;
    mapping(address => uint256) public appealCount;
    
    event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
    event Withdrawn(address indexed user, uint256 amount, uint256 reward);
    event AppealFiled(address indexed user, uint256 slashAmount, string reason);
    event AppealResolved(address indexed user, bool accepted);
    
    constructor(address _stakingToken) {
        stakingToken = IERC20(_stakingToken);
        // Initialize lock period multipliers
        lockPeriodMultipliers[30] = 100;   // 1.0x for 30 days
        lockPeriodMultipliers[90] = 125;   // 1.25x for 90 days
        lockPeriodMultipliers[180] = 150;  // 1.5x for 180 days
    }
    
    // Add validation modifiers
    modifier validateLockPeriod(uint256 lockPeriod) {
        require(lockPeriod >= MIN_LOCK_PERIOD, "Lock period too short");
        require(lockPeriod <= MAX_LOCK_PERIOD, "Lock period too long");
        require(lockPeriodMultipliers[lockPeriod] > 0, "Invalid lock period");
        _;
    }
    
    modifier enforceStakingCooldown() {
        require(
            block.timestamp >= lastStakeTime[msg.sender] + COOLDOWN_PERIOD,
            "Must wait cooldown period"
        );
        _;
    }

    function stake(uint256 amount, uint256 lockPeriod) 
        external 
        nonReentrant 
        whenNotPaused
        validateLockPeriod(lockPeriod)
        enforceStakingCooldown 
    {
        require(amount >= MIN_STAKE, "Below minimum stake amount");
        require(amount <= stakingToken.balanceOf(msg.sender), "Insufficient balance");
        
        // Update cooldown
        lastStakeTime[msg.sender] = block.timestamp;
        
        // Transfer tokens
        require(
            stakingToken.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );
        
        stakes[msg.sender] = Stake({
            amount: amount,
            startTime: block.timestamp,
            lockPeriod: lockPeriod,
            rewardMultiplier: lockPeriodMultipliers[lockPeriod]
        });
        
        emit Staked(msg.sender, amount, lockPeriod);
    }
    
    function withdraw() external nonReentrant whenNotPaused {
        Stake memory userStake = stakes[msg.sender];
        require(userStake.amount > 0, "No active stake");
        require(block.timestamp >= userStake.startTime + userStake.lockPeriod * 1 days, 
                "Lock period not ended");
        
        uint256 reward = calculateReward(msg.sender);
        uint256 total = userStake.amount + reward;
        
        delete stakes[msg.sender];
        
        stakingToken.transfer(msg.sender, total);
        emit Withdrawn(msg.sender, userStake.amount, reward);
    }
    
    function calculateReward(address user) public view returns (uint256) {
        Stake memory userStake = stakes[user];
        if (userStake.amount == 0) return 0;
        
        uint256 timeStaked = block.timestamp - userStake.startTime;
        uint256 baseReward = userStake.amount * BASE_APR * userStake.rewardMultiplier * timeStaked / 
                            (365 days * 10000 * 100);
        
        // Get resource contribution multiplier from ResourceManager
        ResourceManager resourceManager = ResourceManager(resourceManagerAddress);
        uint256 resourceMultiplier = resourceManager.calculateFairShare(user);
        
        return baseReward * (100 + resourceMultiplier) / 100;
    }

    function fileAppeal(string calldata reason) external payable {
        require(msg.value >= APPEAL_FEE, "Insufficient appeal fee");
        require(stakes[msg.sender].amount > 0, "No active stake");
        require(appealCount[msg.sender] < MAX_APPEALS_PER_USER, "Too many appeals");
        require(bytes(reason).length <= 200, "Reason too long");
        
        appealCount[msg.sender]++;
        
        appeals[msg.sender].push(Appeal({
            slashAmount: stakes[msg.sender].amount,
            timestamp: block.timestamp,
            reason: reason,
            resolved: false,
            accepted: false
        }));
        
        emit AppealFiled(msg.sender, stakes[msg.sender].amount, reason);
    }
    
    function resolveAppeal(address user, uint256 appealId, bool accepted) external onlyOwner {
        Appeal storage appeal = appeals[user][appealId];
        require(!appeal.resolved, "Appeal already resolved");
        require(block.timestamp <= appeal.timestamp + APPEAL_WINDOW, "Appeal window closed");
        
        appeal.resolved = true;
        appeal.accepted = accepted;
        
        if (accepted) {
            // Restore slashed amount
            stakes[user].amount += appeal.slashAmount;
        }
        
        emit AppealResolved(user, accepted);
    }

    // Add view functions for frontend
    function getStakeInfo(address user) external view returns (
        uint256 amount,
        uint256 startTime,
        uint256 lockPeriod,
        uint256 rewardMultiplier
    ) {
        Stake memory stake = stakes[user];
        return (
            stake.amount,
            stake.startTime,
            stake.lockPeriod,
            stake.rewardMultiplier
        );
    }
    
    function getAppealCount(address user) external view returns (uint256) {
        return appealCount[user];
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }
}
