// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract VAInToken is ERC20Votes, Pausable, Ownable {
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 1e18; // 100M tokens
    uint256 public constant MAX_SUPPLY = 200_000_000 * 1e18; // 200M max supply
    
    // Distribution pools
    uint256 public trainingRewardsPool;    // 40%
    uint256 public developmentFund;        // 20%
    uint256 public communityTreasury;      // 20%
    uint256 public teamAndAdvisors;        // 15%
    uint256 public liquidityPool;          // 5%
    
    uint256 public emissionRate;
    uint256 public lastEmissionUpdate;
    
    mapping(address => uint256) public vestingBalances;
    mapping(address => uint256) public vestingStart;
    
    event EmissionRateUpdated(uint256 newRate);
    event RewardsDistributed(address indexed to, uint256 amount);
    
    // Constants for validation
    uint256 public constant MIN_VESTING_DURATION = 180 days; // 6 months minimum
    uint256 public constant MAX_EMISSION_RATE = 1000; // 10% max emission rate
    uint256 public constant REWARDS_THRESHOLD = 1000 * 1e18; // Min reward distribution

    // Additional events for monitoring
    event VestingScheduleCreated(address indexed beneficiary, uint256 amount, uint256 startTime);
    event VestingReleased(address indexed beneficiary, uint256 amount);
    event PoolBalanceUpdated(string poolName, uint256 newBalance);
    event EmergencyWithdraw(address indexed to, uint256 amount);

    // Additional validation constants
    uint256 public constant MAX_BATCH_SIZE = 100;
    uint256 public constant MIN_TRANSFER_AMOUNT = 100 * 1e18; // 100 tokens minimum
    uint256 public constant DAILY_REWARD_LIMIT = 1_000_000 * 1e18; // 1M daily limit
    
    // Tracking
    uint256 public dailyRewardsDistributed;
    uint256 public lastRewardResetTime;
    mapping(address => bool) public isPoolManager;
    mapping(address => uint256) public vestingReleaseTime;
    
    event PoolManagerUpdated(address indexed manager, bool status);
    event BatchVestingCreated(address[] beneficiaries, uint256[] amounts);
    event PoolAllocationUpdated(string poolName, uint256 oldAmount, uint256 newAmount);
    event DailyLimitReset(uint256 timestamp);

    // Add tracking for operations
    mapping(address => uint256) public lastOperationTime;
    mapping(address => uint256) public operationCount;
    
    // Add validation constants  
    uint256 public constant OPERATION_COOLDOWN = 1 hours;
    uint256 public constant MAX_OPERATIONS_PER_DAY = 10;
    uint256 public constant MAX_VESTING_AMOUNT = 1_000_000 * 1e18; // 1M tokens
    
    modifier enforceRateLimit() {
        require(
            block.timestamp >= lastOperationTime[msg.sender] + OPERATION_COOLDOWN,
            "Please wait before next operation"
        );
        require(
            operationCount[msg.sender] < MAX_OPERATIONS_PER_DAY,
            "Daily operation limit reached"
        );
        _;
        
        lastOperationTime[msg.sender] = block.timestamp;
        operationCount[msg.sender]++;
    }

    modifier onlyPoolManager() {
        require(isPoolManager[msg.sender] || owner() == msg.sender, "Not authorized");
        _;
    }

    constructor() 
        ERC20("vAIn Token", "VAIN")
        ERC20Permit("vAIn Token") 
    {
        require(INITIAL_SUPPLY <= MAX_SUPPLY, "Initial supply exceeds max supply");
        _mint(address(this), INITIAL_SUPPLY);
        
        // Initialize pools with validation
        require(
            INITIAL_SUPPLY * 40 / 100 + 
            INITIAL_SUPPLY * 20 / 100 +
            INITIAL_SUPPLY * 20 / 100 +
            INITIAL_SUPPLY * 15 / 100 +
            INITIAL_SUPPLY * 5 / 100 == INITIAL_SUPPLY,
            "Pool allocations must sum to total supply"
        );
        
        // Initialize pools
        trainingRewardsPool = INITIAL_SUPPLY * 40 / 100;
        developmentFund = INITIAL_SUPPLY * 20 / 100;
        communityTreasury = INITIAL_SUPPLY * 20 / 100;
        teamAndAdvisors = INITIAL_SUPPLY * 15 / 100;
        liquidityPool = INITIAL_SUPPLY * 5 / 100;
        
        lastEmissionUpdate = block.timestamp;

        emit PoolBalanceUpdated("trainingRewardsPool", trainingRewardsPool);
        emit PoolBalanceUpdated("developmentFund", developmentFund);
        emit PoolBalanceUpdated("communityTreasury", communityTreasury);
        emit PoolBalanceUpdated("teamAndAdvisors", teamAndAdvisors);
        emit PoolBalanceUpdated("liquidityPool", liquidityPool);
    }
    
    function createVestingSchedule(
        address beneficiary, 
        uint256 amount
    ) 
        external 
        onlyOwner 
        enforceRateLimit 
    {
        require(beneficiary != address(0), "Invalid beneficiary address");
        require(amount > 0 && amount <= MAX_VESTING_AMOUNT, "Invalid amount");
        require(vestingBalances[beneficiary] == 0, "Vesting already exists");
        require(
            teamAndAdvisors >= amount,
            "Insufficient team allocation"
        );
        
        vestingBalances[beneficiary] = amount;
        vestingStart[beneficiary] = block.timestamp;
        vestingReleaseTime[beneficiary] = block.timestamp + MIN_VESTING_DURATION;
        
        teamAndAdvisors -= amount;
        
        emit VestingScheduleCreated(beneficiary, amount, block.timestamp);
        emit PoolBalanceUpdated("teamAndAdvisors", teamAndAdvisors);
    }

    function releaseVesting() external {
        require(vestingBalances[msg.sender] > 0, "No vesting balance");
        require(block.timestamp >= vestingStart[msg.sender] + MIN_VESTING_DURATION, "Vesting period not complete");
        
        uint256 amount = vestingBalances[msg.sender];
        vestingBalances[msg.sender] = 0;
        _transfer(address(this), msg.sender, amount);
        
        emit VestingReleased(msg.sender, amount);
    }

    function distributeRewards(address to, uint256 amount) external onlyOwner whenNotPaused {
        require(to != address(0), "Invalid reward recipient");
        require(amount >= REWARDS_THRESHOLD, "Below minimum reward threshold");
        require(amount <= trainingRewardsPool, "Exceeds rewards pool");
        
        // Check and update daily limits
        if(block.timestamp >= lastRewardResetTime + 1 days) {
            dailyRewardsDistributed = 0;
            lastRewardResetTime = block.timestamp;
            emit DailyLimitReset(block.timestamp);
        }
        
        require(dailyRewardsDistributed + amount <= DAILY_REWARD_LIMIT, "Daily limit exceeded");
        dailyRewardsDistributed += amount;

        trainingRewardsPool -= amount;
        _transfer(address(this), to, amount);
        
        emit RewardsDistributed(to, amount);
        emit PoolBalanceUpdated("trainingRewardsPool", trainingRewardsPool);
    }
    
    function updateEmissionRate(uint256 _newRate) external onlyOwner {
        require(_newRate <= MAX_EMISSION_RATE, "Emission rate too high");
        require(_newRate != emissionRate, "Same emission rate");
        
        emissionRate = _newRate;
        lastEmissionUpdate = block.timestamp;
        
        emit EmissionRateUpdated(_newRate);
    }

    function setPoolManager(address manager, bool status) external onlyOwner {
        require(manager != address(0), "Invalid manager address");
        isPoolManager[manager] = status;
        emit PoolManagerUpdated(manager, status);
    }

    function batchCreateVesting(
        address[] calldata beneficiaries,
        uint256[] calldata amounts
    ) external onlyPoolManager {
        require(beneficiaries.length == amounts.length, "Length mismatch");
        require(beneficiaries.length <= MAX_BATCH_SIZE, "Batch too large");
        
        uint256 totalAmount;
        for(uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        require(totalAmount <= teamAndAdvisors, "Exceeds available allocation");
        
        for(uint256 i = 0; i < beneficiaries.length; i++) {
            require(beneficiaries[i] != address(0), "Invalid beneficiary");
            require(amounts[i] > 0, "Invalid amount");
            require(vestingBalances[beneficiaries[i]] == 0, "Already vesting");
            
            vestingBalances[beneficiaries[i]] = amounts[i];
            vestingStart[beneficiaries[i]] = block.timestamp;
            vestingReleaseTime[beneficiaries[i]] = block.timestamp + MIN_VESTING_DURATION;
        }
        
        teamAndAdvisors -= totalAmount;
        emit BatchVestingCreated(beneficiaries, amounts);
        emit PoolBalanceUpdated("teamAndAdvisors", teamAndAdvisors);
    }

    function updatePoolAllocation(
        string calldata poolName,
        uint256 newAmount
    ) external onlyPoolManager whenNotPaused {
        require(newAmount <= INITIAL_SUPPLY, "Exceeds initial supply");
        
        uint256 oldAmount;
        if (keccak256(bytes(poolName)) == keccak256(bytes("trainingRewardsPool"))) {
            oldAmount = trainingRewardsPool;
            trainingRewardsPool = newAmount;
        } else if (keccak256(bytes(poolName)) == keccak256(bytes("developmentFund"))) {
            oldAmount = developmentFund;
            developmentFund = newAmount;
        } // ...add other pools similarly
        
        emit PoolAllocationUpdated(poolName, oldAmount, newAmount);
        emit PoolBalanceUpdated(poolName, newAmount);
    }

    // Add view functions for frontend
    function getVestingInfo(address beneficiary) external view returns (
        uint256 balance,
        uint256 startTime,
        uint256 releaseTime,
        bool isVested
    ) {
        return (
            vestingBalances[beneficiary],
            vestingStart[beneficiary],
            vestingReleaseTime[beneficiary],
            block.timestamp >= vestingReleaseTime[beneficiary]
        );
    }
    
    function getDailyLimits(address user) external view returns (
        uint256 operationsToday,
        uint256 operationsRemaining,
        uint256 cooldownEnds
    ) {
        return (
            operationCount[user],
            MAX_OPERATIONS_PER_DAY - operationCount[user],
            lastOperationTime[user] + OPERATION_COOLDOWN
        );
    }

    // Emergency Functions
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function emergencyWithdraw(address to, uint256 amount) external onlyOwner whenPaused {
        require(to != address(0), "Invalid withdrawal address");
        require(amount <= balanceOf(address(this)), "Insufficient contract balance");
        
        _transfer(address(this), to, amount);
        emit EmergencyWithdraw(to, amount);
    }
    
    // Override required by Solidity
    function _afterTokenTransfer(address from, address to, uint256 amount)
        internal
        override(ERC20Votes)
    {
        super._afterTokenTransfer(from, to, amount);
    }

    function _mint(address to, uint256 amount)
        internal
        override(ERC20Votes)
    {
        super._mint(to, amount);
    }

    function _burn(address account, uint256 amount)
        internal
        override(ERC20Votes)
    {
        super._burn(account, amount);
    }
}
