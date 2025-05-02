#!/usr/bin/env python3
"""
Command-line interface for interacting with the monitoring system.
"""
import os
import sys
import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from monitoring import initialize_monitoring, stop_monitoring, get_recent_alerts, get_monitoring_stats

async def show_alerts(args):
    """Display recent alerts"""
    await initialize_monitoring()
    
    alerts = get_recent_alerts(
        level=args.level, 
        resolved=False if args.unresolved else None,
        limit=args.limit
    )
    
    if not alerts:
        print("No alerts found.")
        return
    
    print(f"Recent alerts ({len(alerts)}):")
    print("-" * 80)
    
    for i, alert in enumerate(alerts):
        timestamp = datetime.fromtimestamp(alert["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. [{alert['level'].upper()}] {timestamp} - {alert['message']}")
        print(f"   Source: {alert['source_file']}")
        if alert["resolved"]:
            print(f"   ✓ Resolved: {alert['resolution_action']}")
        print()
    
    await stop_monitoring()

async def show_stats(args):
    """Display monitoring statistics"""
    await initialize_monitoring()
    
    stats = get_monitoring_stats()
    
    print("Monitoring Statistics:")
    print("-" * 80)
    print(f"Uptime: {stats.get('uptime_formatted', 'N/A')}")
    print(f"Total alerts: {stats.get('total_alerts', 0)}")
    print(f"Auto-resolved: {stats.get('auto_resolved', 0)}")
    
    by_level = stats.get('by_level', {})
    print("\nAlerts by level:")
    print(f"  Critical: {by_level.get('critical', 0)}")
    print(f"  Error: {by_level.get('error', 0)}")
    print(f"  Warning: {by_level.get('warning', 0)}")
    print(f"  Info: {by_level.get('info', 0)}")
    
    await stop_monitoring()

def main():
    """Main entry point for the monitor CLI"""
    parser = argparse.ArgumentParser(description="vAIn P2P AGI Monitoring CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Alerts command
    alerts_parser = subparsers.add_parser("alerts", help="Show recent alerts")
    alerts_parser.add_argument("--level", "-l", type=str, choices=["critical", "error", "warning", "info"],
                             help="Filter by alert level")
    alerts_parser.add_argument("--unresolved", "-u", action="store_true", 
                             help="Show only unresolved alerts")
    alerts_parser.add_argument("--limit", "-n", type=int, default=10,
                             help="Maximum number of alerts to show")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show monitoring statistics")
    
    args = parser.parse_args()
    
    if args.command == "alerts":
        asyncio.run(show_alerts(args))
    elif args.command == "stats":
        asyncio.run(show_stats(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()