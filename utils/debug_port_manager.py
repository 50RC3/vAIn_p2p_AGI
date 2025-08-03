import os
import sys
import socket
import logging
import random
from typing import Optional, Tuple, List, Dict, Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class DebugPortManager:
    """
    Utility to manage debug ports and handle port conflicts
    """
    DEFAULT_DEBUG_PORT = 5678
    PORT_RANGE = (5000, 6000)
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is currently in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except socket.error:
                return True
    
    @staticmethod
    def find_process_using_port(port: int) -> Optional[Any]:
        """Find the process that is using the specified port."""
        if not HAS_PSUTIL:
            return None
            
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Get connections separately since it's not a valid attribute for process_iter
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        return proc
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return None
    
    @staticmethod
    def find_available_port(start_range: int = None, end_range: int = None) -> int:
        """Find an available port within the specified range."""
        start = start_range or DebugPortManager.PORT_RANGE[0]
        end = end_range or DebugPortManager.PORT_RANGE[1]
        
        # First check the default port
        if not DebugPortManager.is_port_in_use(DebugPortManager.DEFAULT_DEBUG_PORT):
            return DebugPortManager.DEFAULT_DEBUG_PORT
        
        # Try random ports in the range to avoid sequential conflicts
        used_ports = set()
        while len(used_ports) < (end - start):
            port = random.randint(start, end)
            if port in used_ports:
                continue
                
            used_ports.add(port)
            if not DebugPortManager.is_port_in_use(port):
                return port
                
        raise RuntimeError(f"No available ports found in range {start}-{end}")
    
    @staticmethod
    def kill_process_using_port(port: int, force: bool = False) -> bool:
        """Attempt to kill the process using the specified port."""
        proc = DebugPortManager.find_process_using_port(port)
        if not proc:
            logger.warning(f"No process found using port {port}")
            return False
            
        proc_info = f"{proc.name()} (PID: {proc.pid})"
        
        if not force:
            confirm = input(f"Process {proc_info} is using port {port}. Kill it? (y/N): ")
            if confirm.lower() != 'y':
                return False
                
        try:
            proc.kill()
            proc.wait(timeout=5)
            logger.info(f"Successfully terminated process {proc_info}")
            return True
        except Exception as e:
            if HAS_PSUTIL:
                # Handle psutil-specific exceptions
                if isinstance(e, (psutil.AccessDenied, psutil.NoSuchProcess, psutil.TimeoutExpired)):
                    logger.error(f"Failed to kill process {proc_info}: {str(e)}")
                    return False
            logger.error(f"Failed to kill process {proc_info}: {str(e)}")
            return False
    
    @classmethod
    def get_debug_command(cls, script_path: str, port: Optional[int] = None) -> str:
        """Get the debug command with an available port."""
        if port is None:
            port = cls.find_available_port()
            
        return f"python -m debugpy --listen {port} --wait-for-client {script_path}"
    
    @classmethod
    def handle_port_conflict(cls, port: int = DEFAULT_DEBUG_PORT) -> Dict[str, Any]:
        """Handle a port conflict situation and return resolution options."""
        result = {
            "success": False,
            "original_port": port,
            "message": ""
        }
        
        proc = cls.find_process_using_port(port)
        
        if not proc:
            result["message"] = f"Port {port} appears to be in use, but no process was found using it."
            result["alternative_port"] = cls.find_available_port()
            return result
            
        proc_info = f"{proc.name()} (PID: {proc.pid})"
        result["process_info"] = {
            "pid": proc.pid,
            "name": proc.name(),
            "cmdline": proc.cmdline()
        }
        
        # Try to find an alternative port
        try:
            alternative_port = cls.find_available_port()
            result["alternative_port"] = alternative_port
            result["message"] = (
                f"Port {port} is in use by {proc_info}.\n"
                f"You can use an alternative port: {alternative_port}"
            )
            result["success"] = True
        except Exception as e:
            result["message"] = f"Error finding alternative port: {str(e)}"
            
        return result


def main():
    """Command line interface for the debug port manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Port Manager')
    subparsers = parser.add_subparsers(dest='command')
    
    # Check port command
    check_parser = subparsers.add_parser('check', help='Check if a port is in use')
    check_parser.add_argument('port', type=int, default=DebugPortManager.DEFAULT_DEBUG_PORT, 
                             nargs='?', help='Port to check')
    
    # Find available port command
    find_parser = subparsers.add_parser('find', help='Find an available port')
    find_parser.add_argument('--start', type=int, default=DebugPortManager.PORT_RANGE[0], 
                            help='Start of port range')
    find_parser.add_argument('--end', type=int, default=DebugPortManager.PORT_RANGE[1], 
                            help='End of port range')
    
    # Kill process command
    kill_parser = subparsers.add_parser('kill', help='Kill process using a port')
    kill_parser.add_argument('port', type=int, default=DebugPortManager.DEFAULT_DEBUG_PORT, 
                           nargs='?', help='Port to free')
    kill_parser.add_argument('--force', action='store_true', help='Kill without confirmation')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Start debugging with available port')
    debug_parser.add_argument('script', help='Path to the script to debug')
    debug_parser.add_argument('--port', type=int, help='Preferred debug port')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.command == 'check':
        port = args.port
        in_use = DebugPortManager.is_port_in_use(port)
        if in_use:
            proc = DebugPortManager.find_process_using_port(port)
            if proc:
                print(f"Port {port} is in use by {proc.name()} (PID: {proc.pid})")
            else:
                print(f"Port {port} is in use by an unknown process")
        else:
            print(f"Port {port} is available")
            
    elif args.command == 'find':
        try:
            port = DebugPortManager.find_available_port(args.start, args.end)
            print(f"Available port found: {port}")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    elif args.command == 'kill':
        if DebugPortManager.kill_process_using_port(args.port, args.force):
            print(f"Successfully freed port {args.port}")
        else:
            print(f"Failed to free port {args.port}")
            
    elif args.command == 'debug':
        port = args.port
        
        # If port is specified but in use, handle the conflict
        if port and DebugPortManager.is_port_in_use(port):
            result = DebugPortManager.handle_port_conflict(port)
            print(result["message"])
            
            if "alternative_port" in result:
                port = result["alternative_port"]
                print(f"Using alternative port: {port}")
        
        # If port is still None, find an available one
        if port is None:
            port = DebugPortManager.find_available_port()
            
        cmd = DebugPortManager.get_debug_command(args.script, port)
        print(f"Debug command: {cmd}")
        
        # Execute the command
        try:
            os.system(cmd)
        except Exception as e:
            print(f"Error starting debug session: {str(e)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
