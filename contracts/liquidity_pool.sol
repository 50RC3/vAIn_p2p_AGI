// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract LiquidityPool is ReentrancyGuard, Ownable {
    IERC20 public vainToken;
    mapping(address => uint256) public liquidity;
    uint256 public totalLiquidity;
    
    uint256 public rewardRate;
    uint256 public lastUpdateTime;
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public rewardPerTokenPaid;
    uint256 public rewardPerTokenStored;
    
    event LiquidityAdded(address indexed provider, uint256 amount);
    event LiquidityRemoved(address indexed provider, uint256 amount);
    
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
    
    function addLiquidity(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Amount must be positive");
        vainToken.transferFrom(msg.sender, address(this), amount);
        liquidity[msg.sender] += amount;
        totalLiquidity += amount;
        emit LiquidityAdded(msg.sender, amount);
    }
    
    function removeLiquidity(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Amount must be positive");
        require(liquidity[msg.sender] >= amount, "Insufficient balance");
        
        liquidity[msg.sender] -= amount;
        totalLiquidity -= amount;
        vainToken.transfer(msg.sender, amount);
        
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
        rewardRate = _rewardRate;
    }
}
