from web3 import Web3
from typing import Dict, List
import json
import os

class RewardDistributor:
    def __init__(self, config):
        self.web3 = Web3(Web3.HTTPProvider(config.eth_rpc_url))
        self.token_contract = self._load_contract('vAInToken')
        self.min_contribution = config.min_contribution
        
    def calculate_rewards(self, contributions: Dict[str, float]) -> Dict[str, int]:
        total_contribution = sum(contributions.values())
        rewards = {}
        
        for address, contribution in contributions.items():
            if contribution < self.min_contribution:
                continue
            reward = int((contribution / total_contribution) * self.token_contract.rewards_pool())
            rewards[address] = reward
            
        return rewards
        
    def distribute_rewards(self, rewards: Dict[str, int]):
        for address, amount in rewards.items():
            try:
                tx = self.token_contract.functions.transfer(address, amount).buildTransaction({
                    'from': self.web3.eth.defaultAccount,
                    'nonce': self.web3.eth.getTransactionCount(self.web3.eth.defaultAccount)
                })
                signed_tx = self.web3.eth.account.signTransaction(tx, os.getenv('PRIVATE_KEY'))
                tx_hash = self.web3.eth.sendRawTransaction(signed_tx.rawTransaction)
                self.web3.eth.waitForTransactionReceipt(tx_hash)
            except Exception as e:
                print(f"Failed to distribute rewards to {address}: {e}")
