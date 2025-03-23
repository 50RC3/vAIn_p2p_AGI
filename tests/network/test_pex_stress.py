import unittest
import asyncio
import time
import random
import string
from network.pex import PeerExchange
from tqdm import tqdm

def generate_peer_id():
    """Generate valid peer ID for testing"""
    letters = ''.join(random.choices(string.ascii_letters, k=4))
    numbers = ''.join(random.choices(string.digits, k=4))
    rest = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{letters}{numbers}{rest}"

class TestPeerExchangeStress(unittest.TestCase):
    def setUp(self):
        self.pex = PeerExchange(
            max_peers=1000,
            interactive=False,
            auto_cleanup=True,
            cleanup_threshold=0.8
        )
        self.test_peers = [generate_peer_id() for _ in range(1200)]

    async def test_mass_peer_addition(self):
        """Test adding large numbers of peers"""
        start_time = time.time()
        added = 0
        
        for peer_id in tqdm(self.test_peers, desc="Adding peers"):
            await self.pex.add_peer_interactive(peer_id)
            added += 1
            
            # Verify we never exceed max_peers
            self.assertLessEqual(len(self.pex.peers), self.pex.max_peers)
            
        end_time = time.time()
        print(f"\nAdded {added} peers in {end_time - start_time:.2f} seconds")
        print(f"Final peer count: {len(self.pex.peers)}")

    async def test_cleanup_performance(self):
        """Test cleanup performance with near-capacity peer list"""
        # Fill to 90% capacity
        target_count = int(self.pex.max_peers * 0.9)
        for peer_id in self.test_peers[:target_count]:
            self.pex.add_peer(peer_id)
            
        # Force some peers to be old
        old_peers = random.sample(list(self.pex.peers.keys()), target_count // 2)
        for peer in old_peers:
            self.pex.peers[peer] = time.time() - 100000  # Make them very old
            
        start_time = time.time()
        self.pex._auto_cleanup()
        end_time = time.time()
        
        print(f"\nCleaned up peers in {end_time - start_time:.2f} seconds")
        print(f"Peers after cleanup: {len(self.pex.peers)}")

    async def test_concurrent_operations(self):
        """Test concurrent peer additions and cleanups"""
        async def add_peers(count):
            for _ in range(count):
                peer_id = generate_peer_id()
                await self.pex.add_peer_interactive(peer_id)
                await asyncio.sleep(0.01)  # Simulate network delay

        async def run_cleanups(count):
            for _ in range(count):
                self.pex._auto_cleanup()
                await asyncio.sleep(0.1)

        start_time = time.time()
        await asyncio.gather(
            add_peers(500),
            add_peers(500),
            run_cleanups(20)
        )
        end_time = time.time()

        print(f"\nConcurrent operations completed in {end_time - start_time:.2f} seconds")
        print(f"Final peer count: {len(self.pex.peers)}")
        
    async def test_get_peers_performance(self):
        """Test get_peers performance with large peer list"""
        # Fill to 80% capacity
        target_count = int(self.pex.max_peers * 0.8)
        for peer_id in self.test_peers[:target_count]:
            self.pex.add_peer(peer_id)
            
        batch_sizes = [10, 50, 100, 200]
        for batch_size in batch_sizes:
            start_time = time.time()
            peers = await self.pex.get_peers_interactive(batch_size)
            end_time = time.time()
            
            print(f"\nGetting {batch_size} peers took {end_time - start_time:.4f} seconds")
            self.assertIsNotNone(peers)
            self.assertLessEqual(len(peers), batch_size)

def run_stress_tests():
    """Run all stress tests"""
    async def run_tests():
        test_case = TestPeerExchangeStress()
        test_case.setUp()
        await test_case.test_mass_peer_addition()
        await test_case.test_cleanup_performance()
        await test_case.test_concurrent_operations()
        await test_case.test_get_peers_performance()

    asyncio.run(run_tests())

if __name__ == '__main__':
    run_stress_tests()
