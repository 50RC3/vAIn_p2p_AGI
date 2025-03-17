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
    
    event EmissionRateUpdated(uint256 newRate);
    event RewardsDistributed(address indexed to, uint256 amount);
    
    constructor() 
        ERC20("vAIn Token", "VAIN")
        ERC20Permit("vAIn Token") 
    {
        _mint(address(this), INITIAL_SUPPLY);
        
        // Initialize pools
        trainingRewardsPool = INITIAL_SUPPLY * 40 / 100;
        developmentFund = INITIAL_SUPPLY * 20 / 100;
        communityTreasury = INITIAL_SUPPLY * 20 / 100;
        teamAndAdvisors = INITIAL_SUPPLY * 15 / 100;
        liquidityPool = INITIAL_SUPPLY * 5 / 100;
        
        lastEmissionUpdate = block.timestamp;
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function distributeRewards(address to, uint256 amount) external onlyOwner whenNotPaused {
        require(amount <= trainingRewardsPool, "Exceeds rewards pool");
        trainingRewardsPool -= amount;
        _transfer(address(this), to, amount);
        emit RewardsDistributed(to, amount);
    }
    
    function updateEmissionRate(uint256 _newRate) external onlyOwner {
        emissionRate = _newRate;
        lastEmissionUpdate = block.timestamp;
        emit EmissionRateUpdated(_newRate);
    }

    // Override required functions
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    function _afterTokenTransfer(address from, address to, uint256 amount) internal override(ERC20Votes) {
        super._afterTokenTransfer(from, to, amount);
    }

    function _mint(address to, uint256 amount) internal override(ERC20Votes) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        super._mint(to, amount);
    }

    function _burn(address account, uint256 amount) internal override(ERC20Votes) {
        super._burn(account, amount);
    }
}
