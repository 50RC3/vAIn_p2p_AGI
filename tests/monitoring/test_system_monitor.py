import unittest
import asyncio
from unittest.mock import patch, MagicMock
import psutil
from monitoring.system_monitor import SystemMonitor, SystemMetrics

class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = SystemMonitor(check_interval=1)

    def tearDown(self):
        asyncio.run(self.monitor.cleanup())

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    async def test_get_metrics(self, mock_net, mock_disk, mock_mem, mock_cpu):
        # Setup mocks
        mock_cpu.return_value = 50.0
        mock_mem.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        mock_net.return_value = MagicMock(
            bytes_sent=1000,
            bytes_recv=2000,
            _asdict=lambda: {'bytes_sent': 1000, 'bytes_recv': 2000}
        )

        metrics = await self.monitor.get_metrics()
        
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.disk_usage, 70.0)
        self.assertEqual(metrics.network_io['bytes_sent'], 1000)

    async def test_error_tracking(self):
        test_error = ValueError("Test error")
        self.monitor.track_error(test_error)
        
        summary = self.monitor.get_error_summary()
        self.assertEqual(summary['ValueError'], 1)

    @patch('psutil.cpu_percent')
    async def test_critical_levels(self, mock_cpu):
        mock_cpu.return_value = 95.0
        metrics = await self.monitor.get_metrics()
        self.assertTrue(self.monitor._check_critical_levels(metrics))

if __name__ == '__main__':
    unittest.main()
