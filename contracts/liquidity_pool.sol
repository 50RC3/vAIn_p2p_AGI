// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract LiquidityPool is ReentrancyGuard, Ownable, Pausable {
    IERC20 public vainToken;
    mapping(address => uint256) public liquidity;
    uint256 public totalLiquidity;
    
    uint256 public rewardRate;
    uint256 public lastUpdateTime;
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public rewardPerTokenPaid;
    uint256 public rewardPerTokenStored;

    // Configuration limits
    uint256 public constant MAX_REWARD_RATE = 1000; // 1000 tokens per second max
    uint256 public constant MIN_LIQUIDITY = 100; // Minimum liquidity amount
    uint256 public constant MAX_LIQUIDITY_PER_USER = 1000000; // Maximum per user
    uint256 public cooldownPeriod = 1 hours; // Time between withdrawals
    
    // Rate limiting
    mapping(address => uint256) public lastWithdrawalTime;
    mapping(address => uint256) public dailyWithdrawalAmount;
    uint256 public constant DAILY_WITHDRAWAL_LIMIT = 100000;
    
    // Events
    event LiquidityAdded(address indexed provider, uint256 amount);
    event LiquidityRemoved(address indexed provider, uint256 amount);
    event RewardRateUpdated(uint256 oldRate, uint256 newRate);
    event EmergencyWithdraw(address indexed user, uint256 amount);
    event ConfigUpdated(string param, uint256 value);
    event PoolPaused(address indexed by, string reason);
    event PoolUnpaused(address indexed by);
    
    constructor(address _vainToken) {
        vainToken = IERC20(_vainToken);
        lastUpdateTime = block.timestamp;
    }
    
    modifier updateReward(address account) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = block.timestamp;
        if (account != address(0)) {
            rewards[account] = earned(account);
            rewardPerTokenPaid[account] = rewardPerTokenStored;
        }
        _;
    }
    
    function rewardPerToken() public view returns (uint256) {
        if (totalLiquidity == 0) return rewardPerTokenStored;
        return rewardPerTokenStored + (
            ((block.timestamp - lastUpdateTime) * rewardRate * 1e18) / totalLiquidity
        );
    }
    
    function earned(address account) public view returns (uint256) {
        return (liquidity[account] * 
            (rewardPerToken() - rewardPerTokenPaid[account]) / 1e18) + 
            rewards[account];
    }
    
    function addLiquidity(uint256 amount) external nonReentrant updateReward(msg.sender) whenNotPaused {
        require(amount > 0, "Amount must be positive");
        require(amount >= MIN_LIQUIDITY, "Below minimum liquidity");
        require(liquidity[msg.sender] + amount <= MAX_LIQUIDITY_PER_USER, "Exceeds max liquidity per user");

        vainToken.transferFrom(msg.sender, address(this), amount);
        liquidity[msg.sender] += amount;
        totalLiquidity += amount;
        emit LiquidityAdded(msg.sender, amount);
    }
    
    function removeLiquidity(uint256 amount) external nonReentrant updateReward(msg.sender) whenNotPaused {
        require(amount > 0, "Amount must be positive");
        require(liquidity[msg.sender] >= amount, "Insufficient balance");
        require(block.timestamp >= lastWithdrawalTime[msg.sender] + cooldownPeriod, "Cooldown period active");
        
        uint256 todayWithdrawals = dailyWithdrawalAmount[msg.sender] + amount;
        require(todayWithdrawals <= DAILY_WITHDRAWAL_LIMIT, "Daily withdrawal limit exceeded");
        
        liquidity[msg.sender] -= amount;
        totalLiquidity -= amount;
        dailyWithdrawalAmount[msg.sender] = todayWithdrawals;
        lastWithdrawalTime[msg.sender] = block.timestamp;
        
        require(vainToken.transfer(msg.sender, amount), "Transfer failed");
        emit LiquidityRemoved(msg.sender, amount);
    }
    
    function getReward() external nonReentrant updateReward(msg.sender) {
        uint256 reward = rewards[msg.sender];
        if (reward > 0) {
            rewards[msg.sender] = 0;
            vainToken.transfer(msg.sender, reward);
        }
    }
    
    function setRewardRate(uint256 _rewardRate) external onlyOwner updateReward(address(0)) {
        require(_rewardRate <= MAX_REWARD_RATE, "Exceeds maximum reward rate");
        uint256 oldRate = rewardRate;
        rewardRate = _rewardRate;
        emit RewardRateUpdated(oldRate, _rewardRate);
    }

    // Emergency functions
    function emergencyWithdraw() external nonReentrant {
        uint256 amount = liquidity[msg.sender];
        require(amount > 0, "No liquidity to withdraw");
        
        liquidity[msg.sender] = 0;
        totalLiquidity -= amount;
        rewards[msg.sender] = 0;
        
        require(vainToken.transfer(msg.sender, amount), "Transfer failed");
        emit EmergencyWithdraw(msg.sender, amount);
    }

    function pause(string calldata reason) external onlyOwner {
        _pause();
        emit PoolPaused(msg.sender, reason);
    }

    function unpause() external onlyOwner {
        _unpause();
        emit PoolUnpaused(msg.sender);
    }

    // Configuration functions
    function setCooldownPeriod(uint256 period) external onlyOwner {
        require(period >= 1 hours && period <= 7 days, "Invalid cooldown period");
        cooldownPeriod = period;
        emit ConfigUpdated("cooldownPeriod", period);
    }

    // View functions
    function getUserLiquidity(address user) external view returns (
        uint256 balance,
        uint256 pendingRewards,
        uint256 nextWithdrawalTime,
        uint256 remainingDailyLimit
    ) {
        balance = liquidity[user];
        pendingRewards = earned(user);
        nextWithdrawalTime = lastWithdrawalTime[user] + cooldownPeriod;
        remainingDailyLimit = DAILY_WITHDRAWAL_LIMIT - dailyWithdrawalAmount[user];
    }
}
