require("@nomiclabs/hardhat-waffle");
require("dotenv").config();

module.exports = {
  solidity: "0.8.17",
  networks: {
    hardhat: {
      mining: {
        auto: false,
        interval: 5000 // Consistent block time for testing
      },
      blockGasLimit: 30000000,
      allowUnlimitedContractSize: true // For complex integration tests
    },
    localhost: {
      url: "http://127.0.0.1:8545",
      timeout: 60000 // Higher timeout for interactive testing
    }
  },
  mocha: {
    timeout: 60000, // Longer timeout for interactive tests
    bail: true // Stop on first test failure
  }
};
