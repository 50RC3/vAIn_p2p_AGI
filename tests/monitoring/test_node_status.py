import unittest
import asyncio
from unittest.mock import patch, MagicMock
import psutil
from monitoring.node_status import NodeMonitor, NodeMetrics
from core.interactive_utils import InteractiveSession

class TestNodeMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = NodeMonitor(sampling_interval=1)
        self.interactive_monitor = NodeMonitor(sampling_interval=1, interactive=True)
        self.non_interactive_monitor = NodeMonitor(sampling_interval=1, interactive=False)

    def tearDown(self):
        self.monitor.cleanup()
        self.interactive_monitor.cleanup()
        self.non_interactive_monitor.cleanup()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_get_metrics_success(self, mock_net, mock_disk, mock_mem, mock_cpu):
        # Setup mocks
        mock_cpu.return_value = 50.0
        mock_mem.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net.return_value = MagicMock(
            bytes_sent=1000,
            bytes_recv=2000
        )

        # Test successful metrics collection
        metrics = self.monitor.get_metrics()
        self.assertIsInstance(metrics, NodeMetrics)
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.disk_usage, 70.0)
        self.assertEqual(metrics.network_io['bytes_sent'], 1000)
        self.assertEqual(metrics.network_io['bytes_recv'], 2000)

    @patch('psutil.cpu_percent', side_effect=psutil.Error("Test error"))
    async def test_get_metrics_interactive_failure(self, mock_cpu):
        # Test error handling in interactive mode
        result = await self.interactive_monitor.get_metrics_interactive()
        self.assertIsNone(result)

    @patch('psutil.cpu_percent')
    @patch('core.interactive_utils.InteractiveSession.confirm_with_timeout')
    async def test_high_cpu_warning(self, mock_confirm, mock_cpu):
        # Test high CPU warning in interactive mode
        mock_cpu.return_value = 95.0
        mock_confirm.return_value = False

        result = await self.interactive_monitor.get_metrics_interactive()
        self.assertIsNone(result)
        mock_confirm.assert_called_once()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_non_interactive_fallback(self, mock_disk, mock_mem, mock_cpu):
        # Test non-interactive fallback behavior
        mock_cpu.side_effect = [psutil.Error("First error"), 50.0]
        mock_mem.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)

        # First call should return None due to error
        result = await self.non_interactive_monitor.get_metrics_interactive()
        self.assertIsNone(result)

        # Second call should succeed after error
        metrics = self.non_interactive_monitor.get_metrics()
        self.assertIsInstance(metrics, NodeMetrics)
        self.assertEqual(metrics.cpu_usage, 50.0)

    @patch('core.interactive_utils.InteractiveSession.save_progress')
    async def test_progress_saving(self, mock_save):
        # Test progress saving functionality
        metrics = NodeMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000}
        )
        
        await self.interactive_monitor._save_progress(metrics.__dict__)
        mock_save.assert_called_once_with(metrics.__dict__)

    async def test_cleanup(self):
        # Test cleanup behavior
        monitor = NodeMonitor()
        monitor.session = MagicMock(spec=InteractiveSession)
        monitor.cleanup()
        monitor.session.cleanup.assert_called_once()

    def test_concurrent_access(self):
        # Test concurrent access to metrics
        async def get_metrics_concurrent():
            tasks = [
                self.monitor.get_metrics_interactive() for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(get_metrics_concurrent())
        self.assertEqual(len(results), 5)

if __name__ == '__main__':
    unittest.main()
