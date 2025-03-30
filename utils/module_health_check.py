import logging
import asyncio
import inspect
import importlib
from typing import Dict, List, Any, Optional, Set
import torch

logger = logging.getLogger(__name__)

class ModuleHealthCheck:
    """Utility for checking the health of modules and their dependencies"""
    
    @staticmethod
    async def check_module(module_path: str) -> Dict[str, Any]:
        """Check a module for common issues"""
        result = {
            "status": "unknown",
            "issues": [],
            "dependencies": {},
            "methods": [],
            "response_handling": {},
            "error_handling": {}
        }
        
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            result["status"] = "imported"
            
            # Check for common classes
            classes = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_path:
                    classes.append(name)
                    
                    # Check if class methods with 'async' are correctly defined
                    methods = inspect.getmembers(obj, predicate=inspect.isfunction)
                    for method_name, method in methods:
                        result["methods"].append(f"{name}.{method_name}")
                        if inspect.iscoroutinefunction(method):
                            # Ensure any method named "_trigger_event" is present
                            if method_name == "_notify_handlers" and not hasattr(obj, "_trigger_event"):
                                result["issues"].append(
                                    f"Class {name} has _notify_handlers but missing _trigger_event (compatibility issue)"
                                )
                            
                            # Check response generation methods
                            if method_name == "_generate_response" or method_name == "process_message":
                                result["response_handling"][f"{name}.{method_name}"] = True
                                # Check for proper error handling
                                source = inspect.getsource(method)
                                if "except Exception" not in source and "except:" not in source:
                                    result["issues"].append(
                                        f"{name}.{method_name} may not have proper exception handling"
                                    )
                                if "ChatResponse" in source:
                                    if "error=" not in source and "error:" not in source:
                                        result["issues"].append(
                                            f"{name}.{method_name} creates ChatResponse but may not set error field"
                                        )
            
            result["classes"] = classes
            
            # Check for torch compatibility
            if hasattr(module, "__file__") and "torch" in module.__file__:
                if torch.cuda.is_available():
                    result["cuda_available"] = True
                    result["cuda_device_count"] = torch.cuda.device_count()
                else:
                    result["cuda_available"] = False
                    if "cuda" in module.__file__:
                        result["issues"].append("Module may require CUDA but it's not available")
            
            # Check for chat pod compatibility
            if "interface" in module_path or "module_integration" in module_path:
                result["chat_pod_compatibility"] = await ModuleHealthCheck._check_chat_pod_compatibility(module)
            
            return result
        
        except ImportError as e:
            result["status"] = "import_failed"
            result["issues"].append(f"Import error: {str(e)}")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Error checking module: {str(e)}")
            return result
    
    @staticmethod
    async def _check_chat_pod_compatibility(module) -> Dict[str, Any]:
        """Check if module is compatible with chat pods"""
        result = {
            "compatible": False,
            "issues": []
        }
        
        try:
            # Look for chat pod specific methods and attributes
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    # Check if this class can handle chat pods
                    if hasattr(obj, "process_message") or hasattr(obj, "_generate_response"):
                        result["compatible"] = True
                        
                        # Check if proper error handling exists in these methods
                        if hasattr(obj, "process_message"):
                            method = getattr(obj, "process_message")
                            source = inspect.getsource(method)
                            if "torch.cuda.OutOfMemoryError" not in source:
                                result["issues"].append(f"{name} doesn't handle GPU out of memory errors")
                        
                        # Check for mobile compatibility for pods
                        if "mobile" in name.lower() or "Mobile" in name:
                            if not hasattr(obj, "_mobile_inference") and not hasattr(obj, "mobile_forward"):
                                result["issues"].append(f"{name} is mobile-related but lacks mobile inference methods")
            
            return result
            
        except Exception as e:
            result["issues"].append(f"Error checking chat pod compatibility: {str(e)}")
            return result
    
    @staticmethod
    async def check_system_dependencies() -> Dict[str, Any]:
        """Check system dependencies and compatibility"""
        result = {
            "python_version": None,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "issues": []
        }
        
        try:
            import sys
            result["python_version"] = sys.version
            
            import torch
            result["torch_version"] = torch.__version__
            result["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                result["cuda_version"] = torch.version.cuda
            else:
                result["issues"].append("CUDA not available, some modules may run slower")
            
            return result
            
        except ImportError as e:
            result["issues"].append(f"Import error: {str(e)}")
            return result
            
        except Exception as e:
            result["issues"].append(f"Error checking dependencies: {str(e)}")
            return result
    
    @staticmethod
    async def check_interface_compatibility() -> Dict[str, Any]:
        """Check interface compatibility between modules"""
        result = {
            "status": "unknown",
            "compatibility_issues": [],
            "interface_methods": {}
        }
        
        try:
            # Get interface classes
            from ai_core.chatbot.interface import ChatbotInterface
            from ai_core.chatbot.mobile_interface import MobileChatInterface
            
            # Check ChatbotInterface
            interface_methods = inspect.getmembers(ChatbotInterface, predicate=inspect.isfunction)
            result["interface_methods"]["ChatbotInterface"] = [method[0] for method in interface_methods]
            
            # Ensure both _notify_handlers and _trigger_event exist (one may be a wrapper for the other)
            if ("_notify_handlers" in result["interface_methods"]["ChatbotInterface"] and
                "_trigger_event" not in result["interface_methods"]["ChatbotInterface"]):
                result["compatibility_issues"].append(
                    "ChatbotInterface has _notify_handlers but missing _trigger_event (needed for backward compatibility)"
                )
                
            # Check MobileChatInterface
            interface_methods = inspect.getmembers(MobileChatInterface, predicate=inspect.isfunction)
            result["interface_methods"]["MobileChatInterface"] = [method[0] for method in interface_methods]
                
            # Check for module integration
            from ai_core.chatbot.module_integration import ModuleIntegration
            integration_methods = inspect.getmembers(ModuleIntegration, predicate=inspect.isfunction)
            result["interface_methods"]["ModuleIntegration"] = [method[0] for method in integration_methods]
            
            # Check that ModuleIntegration can handle both method names
            relay_method = getattr(ModuleIntegration, "_relay_to_interface", None)
            if relay_method:
                source = inspect.getsource(relay_method)
                if "_notify_handlers" in source and "_trigger_event" not in source:
                    result["compatibility_issues"].append(
                        "ModuleIntegration._relay_to_interface only handles _notify_handlers, not _trigger_event"
                    )
            
            # Check for proper response error handling
            for interface in [ChatbotInterface, MobileChatInterface]:
                if hasattr(interface, "_generate_response"):
                    gen_method = getattr(interface, "_generate_response")
                    source = inspect.getsource(gen_method)
                    if "torch.cuda.OutOfMemoryError" not in source:
                        result["compatibility_issues"].append(
                            f"{interface.__name__}._generate_response doesn't handle CUDA out of memory errors"
                        )
            
            # Check for chat pod compatibility
            result["chat_pod_compatibility"] = await ModuleHealthCheck._check_chat_pod_support()
                    
            return result
            
        except ImportError as e:
            result["status"] = "import_failed"
            result["compatibility_issues"].append(f"Import error: {str(e)}")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["compatibility_issues"].append(f"Error checking interfaces: {str(e)}")
            return result
    
    @staticmethod
    async def _check_chat_pod_support() -> Dict[str, Any]:
        """Check for chat pod support in the system"""
        result = {
            "supported": False,
            "issues": []
        }
        
        try:
            # Check if the module integration supports chat pods
            from ai_core.chatbot.module_integration import ModuleIntegration
            
            # Look for chat pod handling methods
            if hasattr(ModuleIntegration, "process_message"):
                # Check if this method can handle pod-specific errors
                method = getattr(ModuleIntegration, "process_message")
                source = inspect.getsource(method)
                
                if "await self._notify_callbacks('error'" in source:
                    result["supported"] = True
                else:
                    result["issues"].append("ModuleIntegration.process_message doesn't notify on errors")
            
            # Look for resource management
            if hasattr(ModuleIntegration, "_check_resources") and hasattr(ModuleIntegration, "_cleanup_resources"):
                result["resource_management"] = True
            else:
                result["issues"].append("ModuleIntegration lacks complete resource management methods")
            
            return result
            
        except ImportError as e:
            result["issues"].append(f"Import error: {str(e)}")
            return result
            
        except Exception as e:
            result["issues"].append(f"Error checking chat pod support: {str(e)}")
            return result

    @staticmethod
    async def check_response_generation() -> Dict[str, Any]:
        """Check for common issues in response generation"""
        result = {
            "status": "unknown",
            "issues": []
        }
        
        try:
            # Import the needed interfaces
            from ai_core.chatbot.interface import ChatbotInterface
            from ai_core.chatbot.mobile_interface import MobileChatInterface
            
            # Check for proper error handling in response generation
            for interface_class in [ChatbotInterface, MobileChatInterface]:
                if hasattr(interface_class, "_generate_response"):
                    method = getattr(interface_class, "_generate_response")
                    source = inspect.getsource(method)
                    
                    # Check if common errors are handled
                    error_types = ["RuntimeError", "ValueError", "TypeError", "AttributeError"]
                    missing_errors = [e for e in error_types if f"except {e}" not in source]
                    
                    if missing_errors:
                        result["issues"].append(
                            f"{interface_class.__name__}._generate_response doesn't handle: {', '.join(missing_errors)}"
                        )
                    
                    # Check if errors are properly returned in the response
                    if "error=" not in source and "error:" not in source:
                        result["issues"].append(
                            f"{interface_class.__name__}._generate_response may not set the error field in responses"
                        )
            
            # Check if the ChatResponse class has the needed fields
            from ai_core.chatbot.interface import ChatResponse
            fields = [f.name for f in inspect.getmembers(ChatResponse) if not f.name.startswith('_')]
            required_fields = ["text", "confidence", "error", "model_version", "latency"]
            
            for field in required_fields:
                if field not in fields:
                    result["issues"].append(f"ChatResponse is missing required field: {field}")
            
            if not result["issues"]:
                result["status"] = "ok"
            else:
                result["status"] = "issues_found"
            
            return result
            
        except ImportError as e:
            result["status"] = "import_failed"
            result["issues"].append(f"Import error: {str(e)}")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Error checking response generation: {str(e)}")
            return result

async def run_health_check():
    """Run a complete health check on the module system"""
    health_check = ModuleHealthCheck()
    
    print("Running system health check...")
    
    # Check system dependencies
    print("\n=== System Dependencies ===")
    deps = await health_check.check_system_dependencies()
    print(f"Python version: {deps['python_version']}")
    print(f"PyTorch version: {deps['torch_version']}")
    print(f"CUDA available: {deps['cuda_available']}")
    if deps['cuda_available']:
        print(f"CUDA version: {deps['cuda_version']}")
    
    if deps['issues']:
        print("\nIssues found:")
        for issue in deps['issues']:
            print(f"- {issue}")
    
    # Check interface compatibility
    print("\n=== Interface Compatibility ===")
    compat = await health_check.check_interface_compatibility()
    
    if compat['compatibility_issues']:
        print("\nCompatibility issues found:")
        for issue in compat['compatibility_issues']:
            print(f"- {issue}")
    else:
        print("No compatibility issues found")
    
    # Check response generation
    print("\n=== Response Generation ===")
    resp_gen = await health_check.check_response_generation()
    print(f"Status: {resp_gen['status']}")
    
    if resp_gen['issues']:
        print("\nIssues found:")
        for issue in resp_gen['issues']:
            print(f"- {issue}")
    else:
        print("No issues found in response generation")
    
    # Check key modules
    modules_to_check = [
        "ai_core.chatbot.interface",
        "ai_core.chatbot.module_integration",
        "ai_core.chatbot.rl_trainer",
        "ai_core.chatbot.mobile_interface",
        "ai_core.resource_management",
        "ai_core.module_registry"
    ]
    
    print("\n=== Module Health ===")
    for module_path in modules_to_check:
        print(f"\nChecking module: {module_path}")
        result = await health_check.check_module(module_path)
        
        print(f"Status: {result['status']}")
        if result['issues']:
            print("Issues:")
            for issue in result['issues']:
                print(f"- {issue}")
        
        # Check chat pod compatibility if available
        if "chat_pod_compatibility" in result:
            pod_compat = result["chat_pod_compatibility"]
            print(f"Chat Pod Compatible: {pod_compat.get('compatible', False)}")
            if pod_compat.get("issues", []):
                print("Chat Pod Issues:")
                for issue in pod_compat["issues"]:
                    print(f"- {issue}")

if __name__ == "__main__":
    asyncio.run(run_health_check())
