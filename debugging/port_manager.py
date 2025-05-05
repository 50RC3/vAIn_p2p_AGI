import socket
import psutil
import logging
import signal
import subprocess
import sys
import os
import platform
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import argparse

from utils.unified_logger import get_logger

logger = get_logger("debug_port_manager")

class DebugPortManager:
    """
    Manages debug ports to prevent conflicts between debugging sessions
    and other processes.
    """
    
    DEFAULT_DEBUG_PORT = 5678
    PORT_RANGE_START = 5670
    PORT_RANGE_END = 5690
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is in use by attempting to bind to it"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except (socket.error, OSError):
                return True
    
    @staticmethod
    def find_available_port(start_port: int = None, max_attempts: int = 20) -> Optional[int]:
        """Find an available port starting from start_port"""
        port = start_port or DebugPortManager.DEFAULT_DEBUG_PORT
        
        for _ in range(max_attempts):
            if not DebugPortManager.is_port_in_use(port):
                return port
            port = port + 1
            if port > DebugPortManager.PORT_RANGE_END:
                port = DebugPortManager.PORT_RANGE_START
        
        logger.error("Failed to find available debug port")
        return None
    
    @staticmethod
    def find_process_using_port(port: int) -> Optional[int]:
        """Find the process ID using a specific port"""
        try:
            if platform.system() == "Windows":
                # Use netstat on Windows
                cmd = f"netstat -ano | findstr :{port}"
                output = subprocess.check_output(cmd, shell=True).decode()
                
                for line in output.strip().split('\n'):
                    if f":{port}" in line and "LISTEN" in line:
                        parts = line.strip().split()
                        if len(parts) > 4:
                            return int(parts[4])
                            
            else:
                # Use lsof on Unix-like systems
                cmd = f"lsof -i :{port} -t"
                output = subprocess.check_output(cmd, shell=True).decode()
                if output.strip():
                    return int(output.strip())
                    
        except (subprocess.SubprocessError, ValueError):
            pass
            
        # Fallback: check all network connections through psutil
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.info['connections']
                    if connections:
                        for conn in connections:
                            if conn.laddr.port == port:
                                return proc.info['pid']
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
            
        return None
    
    @staticmethod
    def kill_process_using_port(port: int, force: bool = False) -> bool:
        """Kill the process using a specific port"""
        pid = DebugPortManager.find_process_using_port(port)
        if not pid:
            logger.warning(f"No process found using port {port}")
            return False
            
        try:
            process = psutil.Process(pid)
            logger.info(f"Killing process {pid} ({process.name()}) using port {port}")
            
            if force:
                process.kill()
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()
                    
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            logger.error(f"Error killing process {pid}: {e}")
            return False
    
    @staticmethod
    def list_debug_processes() -> List[Dict[str, Any]]:
        """List processes in the typical debug port range"""
        result = []
        
        for port in range(DebugPortManager.PORT_RANGE_START, DebugPortManager.PORT_RANGE_END + 1):
            pid = DebugPortManager.find_process_using_port(port)
            if pid:
                try:
                    process = psutil.Process(pid)
                    result.append({
                        "port": port,
                        "pid": pid,
                        "name": process.name(),
                        "cmd": process.cmdline(),
                        "create_time": process.create_time()
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result.append({
                        "port": port,
                        "pid": pid,
                        "name": "unknown",
                        "cmd": [],
                        "create_time": 0
                    })
                    
        return result
    
    @staticmethod
    def release_all_debug_ports(force: bool = False) -> Dict[int, bool]:
        """Release all ports in the debug range"""
        results = {}
        
        for port in range(DebugPortManager.PORT_RANGE_START, DebugPortManager.PORT_RANGE_END + 1):
            if DebugPortManager.is_port_in_use(port):
                results[port] = DebugPortManager.kill_process_using_port(port, force)
                
        return results
    
    @staticmethod
    def reserve_port(port: int = None) -> Optional[int]:
        """Reserve a specific port (or find an available one)"""
        if port is not None:
            # Try to use the requested port
            if DebugPortManager.is_port_in_use(port):
                logger.warning(f"Port {port} is in use, cannot reserve")
                return None
            return port
            
        # Find any available port in range
        return DebugPortManager.find_available_port()

def main():
    """Command line interface for debug port management"""
    parser = argparse.ArgumentParser(description="Debug Port Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Check port
    check_parser = subparsers.add_parser("check", help="Check if a port is in use")
    check_parser.add_argument("port", type=int, help="Port to check")
    
    # Find port
    find_parser = subparsers.add_parser("find", help="Find an available debug port")
    
    # Kill process
    kill_parser = subparsers.add_parser("kill", help="Kill process using a port")
    kill_parser.add_argument("port", type=int, help="Port to free")
    kill_parser.add_argument("-f", "--force", action="store_true", help="Force kill")
    
    # List processes
    subparsers.add_parser("list", help="List processes using debug ports")
    
    # Release all
    release_parser = subparsers.add_parser("release-all", help="Release all debug ports")
    release_parser.add_argument("-f", "--force", action="store_true", help="Force kill")
    
    args = parser.parse_args()
    
    if args.command == "check":
        in_use = DebugPortManager.is_port_in_use(args.port)
        print(f"Port {args.port} is {'in use' if in_use else 'available'}")
        if in_use:
            pid = DebugPortManager.find_process_using_port(args.port)
            if pid:
                try:
                    process = psutil.Process(pid)
                    print(f"Used by process {pid} ({process.name()})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"Used by process {pid}")
                    
    elif args.command == "find":
        port = DebugPortManager.find_available_port()
        if port:
            print(f"Available port: {port}")
        else:
            print("Failed to find available port")
            
    elif args.command == "kill":
        success = DebugPortManager.kill_process_using_port(args.port, args.force)
        print(f"{'Successfully' if success else 'Failed to'} kill process using port {args.port}")
        
    elif args.command == "list":
        processes = DebugPortManager.list_debug_processes()
        if processes:
            print(f"Found {len(processes)} processes using debug ports:")
            for p in processes:
                cmd = " ".join(p['cmd'][:2]) + "..." if len(p['cmd']) > 2 else " ".join(p['cmd'])
                print(f"Port {p['port']}: PID {p['pid']} - {p['name']} - {cmd}")
        else:
            print("No processes found using debug ports")
            
    elif args.command == "release-all":
        results = DebugPortManager.release_all_debug_ports(args.force)
        if results:
            print(f"Released {sum(results.values())} debug ports")
            for port, success in results.items():
                print(f"Port {port}: {'Released' if success else 'Failed'}")
        else:
            print("No debug ports in use")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()