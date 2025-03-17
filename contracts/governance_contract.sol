// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorSettings.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";

contract VAINGovernor is Governor, GovernorSettings, GovernorVotes {
    constructor(IVotes _token)
        Governor("VAIN Governor")
        GovernorSettings(1, 45818, 0)
        GovernorVotes(_token)
    {}

    function votingDelay() public view override returns (uint256) {
        return super.votingDelay();
    }

    function votingPeriod() public view override returns (uint256) {
        return super.votingPeriod();
    }

    function quorum(uint256) public pure override returns (uint256) {
        return 4000e18;
    }
}
