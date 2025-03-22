from web3 import Web3
from typing import Dict, List
import json
import os
import logging
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import InteractionLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardDistributor:
    def __init__(self, config):
        self.web3 = Web3(Web3.HTTPProvider(config.eth_rpc_url))
        self.token_contract = self._load_contract('vAInToken')
        self.min_contribution = config.min_contribution
        self.session = InteractiveSession(
            level=InteractionLevel.NORMAL,
            config=InteractiveConfig(timeout=300, safe_mode=True)
        )
        
    def calculate_rewards(self, contributions: Dict[str, float]) -> Dict[str, int]:
        total_contribution = sum(contributions.values())
        rewards = {}
        
        for address, contribution in contributions.items():
            if contribution < self.min_contribution:
                continue
            reward = int((contribution / total_contribution) * self.token_contract.rewards_pool())
            rewards[address] = reward
            
        return rewards
        
    async def distribute_rewards(self, rewards: Dict[str, int]):
        """Distribute rewards with interactive monitoring."""
        try:
            print("\nStarting Reward Distribution")
            print("=" * 50)
            print(f"Total Recipients: {len(rewards)}")
            
            with tqdm(total=len(rewards)) as pbar:
                for address, amount in rewards.items():
                    try:
                        # Build and sign transaction
                        tx = self.token_contract.functions.transfer(
                            address, amount
                        ).buildTransaction({
                            'from': self.web3.eth.defaultAccount,
                            'nonce': self.web3.eth.getTransactionCount(
                                self.web3.eth.defaultAccount
                            )
                        })
                        
                        # Confirm large transfers
                        if amount > self.token_contract.functions.totalSupply().call() * 0.01:
                            if not await self.session.confirm_with_timeout(
                                f"\nLarge transfer ({amount} tokens) to {address}. Proceed?",
                                timeout=30
                            ):
                                continue

                        signed_tx = self.web3.eth.account.signTransaction(
                            tx, os.getenv('PRIVATE_KEY')
                        )
                        tx_hash = self.web3.eth.sendRawTransaction(
                            signed_tx.rawTransaction
                        )
                        self.web3.eth.waitForTransactionReceipt(tx_hash)
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Failed to distribute rewards to {address}: {e}")
                        if not await self.session.confirm_with_timeout(
                            "\nContinue with remaining distributions? (Y/n): ",
                            timeout=30,
                            default=True
                        ):
                            break

        except Exception as e:
            logger.error(f"Distribution failed: {str(e)}")
            raise
        finally:
            await self.session.cleanup()
