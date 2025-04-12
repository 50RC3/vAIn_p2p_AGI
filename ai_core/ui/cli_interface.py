import asyncio
import logging
import os
import sys
from typing import Optional, Any, Dict, List
import time

from ai_core.chatbot.interface import ChatbotInterface, UIEvent

logger = logging.getLogger(__name__)

class CommandLineInterface:
    """Simple command-line interface for the chatbot."""
    
    def __init__(self, chatbot: ChatbotInterface, resource_monitor=None):
        self.chatbot = chatbot
        self.running = False
        self._task = None
        self._response_processed = set()  # Track processed response hashes
        self.resource_monitor = resource_monitor
        self.status_indicators = {
            'warning': '⚠️',
            'error': '❌',
            'success': '✅',
            'info': 'ℹ️'
        }
        self.last_status_update = time.time()
        self.status_update_interval = 60  # seconds
        
    async def subscribe_to_events(self):
        """Subscribe to chatbot events."""
        await self.chatbot.subscribe('response_generated', self._handle_response)
        await self.chatbot.subscribe('error', self._handle_error)
        await self.chatbot.subscribe('learning_progress', self._handle_learning_progress)
        await self.chatbot.subscribe('resource_warning', self._handle_resource_warning)
        await self.chatbot.subscribe('model_saved', self._handle_model_saved)
        
        logger.info("CLI subscribed to chatbot events")
    
    async def _handle_response(self, data: Dict[str, Any]) -> None:
        """Handle response events from chatbot."""
        # This event handler is only used for background processing
        # We won't display responses here to avoid duplication
        pass
    
    async def _handle_error(self, data: Any) -> None:
        """Handle error events from chatbot."""
        error_msg = data.get('error', 'Unknown error') if isinstance(data, dict) else str(data)
        print(f"\n{self.status_indicators['error']} Error: {error_msg}")
    
    async def _handle_learning_progress(self, data: Any) -> None:
        """Handle learning progress events."""
        if isinstance(data, dict) and 'progress' in data:
            progress = data['progress']
            print(f"\n{self.status_indicators['info']} Learning progress: {progress:.1f}%")
    
    async def _handle_resource_warning(self, data: Any) -> None:
        """Handle resource warning events."""
        if isinstance(data, dict):
            resource = data.get('resource', 'unknown')
            usage = data.get('usage', 0)
            threshold = data.get('threshold', 0)
            print(f"\n{self.status_indicators['warning']} Resource warning: {resource} at {usage:.1f}% (threshold: {threshold:.1f}%)")
    
    async def _handle_model_saved(self, data: Any) -> None:
        """Handle model saved events."""
        if isinstance(data, dict) and 'path' in data:
            path = data['path']
            print(f"\n{self.status_indicators['success']} Model saved to: {path}")
    
    async def _check_system_status(self) -> None:
        """Periodically check system status and display if needed."""
        if not self.resource_monitor:
            return
            
        current_time = time.time()
        if current_time - self.last_status_update >= self.status_update_interval:
            self.last_status_update = current_time
            
            # Get current resource metrics
            try:
                metrics = self.resource_monitor.get_metrics()
                
                # Only show if we're exceeding thresholds
                if metrics.memory_usage > 80 or metrics.cpu_usage > 80:
                    print(f"\n{self.status_indicators['info']} System status - "
                          f"CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%")
            except Exception as e:
                logger.debug(f"Error checking system status: {e}")
    
    async def start(self) -> None:
        """Start the command-line interface."""
        self.running = True
        
        try:
            # Start a new session
            session_id = await self.chatbot.start_session()
            print(f"Started new chat session: {session_id}")
            print("Type 'exit', 'quit', or Ctrl+C to end the session.")
            print("Type 'clear' to clear the current session.")
            print("Type 'status' to see system status.")
            print("-" * 50)
            
            await self.subscribe_to_events()
            
            # Main interaction loop
            while self.running:
                try:
                    # Periodically check system status
                    await self._check_system_status()
                    
                    # Get user input
                    print("\nYou: ", end="", flush=True)
                    user_input = await self._async_input()
                    
                    # Handle special commands
                    if user_input.lower() in ('exit', 'quit'):
                        break
                    elif user_input.lower() == 'clear':
                        await self.chatbot.clear_session()
                        session_id = await self.chatbot.start_session()
                        print(f"Started new chat session: {session_id}")
                        continue
                    elif user_input.lower() == 'status':
                        await self._show_status()
                        continue
                    
                    if not user_input.strip():
                        continue
                    
                    # Process user message
                    print("Processing...", end="\r", flush=True)
                    try:
                        response = await self.chatbot.process_message(user_input)
                        
                        # Generate response hash to avoid duplication
                        response_hash = hash((user_input, response.text))
                        
                        # Clear the "Processing..." text and any artifacts
                        print("\r" + " " * 30 + "\r", end="", flush=True)
                        
                        # Only display the response if there's no error
                        if not response.error:
                            print(f"Bot: {response.text}")
                            
                            # Add to chat history
                            if hasattr(self.chatbot, 'history') and response_hash not in self._response_processed:
                                self.chatbot.history.append((user_input, response.text))
                            
                            # Avoid duplicate feedback requests by tracking processed responses
                            if response_hash not in self._response_processed:
                                self._response_processed.add(response_hash)
                                
                                # Prompt for feedback
                                print("\nRate this response (0-1, or skip with Enter):")
                                feedback = await self._async_input()
                                if feedback.strip() and feedback.replace('.', '', 1).isdigit():
                                    score = float(feedback)
                                    if 0 <= score <= 1:
                                        await self.chatbot.store_feedback(response.id, score)
                                        print(f"Feedback recorded: {score}")
                                    else:
                                        print("Invalid score. Must be between 0 and 1.")
                        else:
                            print(f"Error: {response.error}")
                            
                    except Exception as e:
                        logger.exception("Error processing message")
                        print(f"\nError processing message: {str(e)}")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception("Interface error")
                    print(f"\nError: {e}")
            
        finally:
            # Clean up
            await self.chatbot.clear_session()
            print("\nSession ended.")
            self.running = False
    
    async def _show_status(self) -> None:
        """Show current system status."""
        print("\n----- System Status -----")
        
        # Show chatbot status
        learning_enabled = getattr(self.chatbot, 'learning_enabled', False)
        print(f"Learning enabled: {learning_enabled}")
        
        # Show number of messages in history
        history_size = len(getattr(self.chatbot, 'history', []))
        print(f"Message history: {history_size} items")
        
        # Show feedback statistics
        feedback_count = len(getattr(self.chatbot, 'feedback_scores', []))
        if feedback_count > 0:
            avg_score = sum(item.get('score', 0) for item in self.chatbot.feedback_scores) / feedback_count
            print(f"Feedback: {feedback_count} ratings (avg: {avg_score:.2f})")
        else:
            print("Feedback: No ratings yet")
        
        # Show resource metrics if available
        if self.resource_monitor:
            try:
                metrics = self.resource_monitor.get_metrics()
                print(f"CPU usage: {metrics.cpu_usage:.1f}%")
                print(f"Memory usage: {metrics.memory_usage:.1f}%")
                if metrics.gpu_usage is not None:
                    print(f"GPU usage: {metrics.gpu_usage:.1f}%")
            except Exception as e:
                logger.debug(f"Error getting resource metrics: {e}")
                print("Resource metrics unavailable")
                
        print("-" * 25)
    
    async def stop(self) -> None:
        """Stop the interface."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    @staticmethod
    async def _async_input() -> str:
        """Get user input asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(None, input)

async def run_cli(model, storage, interactive=True, resource_monitor=None):
    """Run the command-line interface."""
    # Create the chatbot interface
    chatbot = ChatbotInterface(model, storage, interactive=interactive)
    
    # Create and start the CLI
    cli = CommandLineInterface(chatbot, resource_monitor)
    try:
        await cli.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await cli.stop()
        await chatbot.cleanup()
