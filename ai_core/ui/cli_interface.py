import asyncio
import logging
import os
import sys
from typing import Optional, Any, Dict

from ai_core.chatbot.interface import ChatbotInterface, UIEvent

logger = logging.getLogger(__name__)

class CommandLineInterface:
    """Simple command-line interface for the chatbot."""
    
    def __init__(self, chatbot: ChatbotInterface):
        self.chatbot = chatbot
        self.running = False
        self._task = None
        self._response_processed = set()  # Track processed response hashes
        
    async def subscribe_to_events(self):
        """Subscribe to chatbot events."""
        await self.chatbot.subscribe('response_generated', self._handle_response)
        await self.chatbot.subscribe('error', self._handle_error)
        
    async def _handle_response(self, data: Dict[str, Any]) -> None:
        """Handle response events from chatbot."""
        # This event handler is only used for background processing
        # We won't display responses here to avoid duplication
        pass
    
    async def _handle_error(self, data: Any) -> None:
        """Handle error events from chatbot."""
        error_msg = data.get('error', 'Unknown error') if isinstance(data, dict) else str(data)
        print(f"\nError: {error_msg}")
    
    async def start(self) -> None:
        """Start the command-line interface."""
        self.running = True
        
        try:
            # Start a new session
            session_id = await self.chatbot.start_session()
            print(f"Started new chat session: {session_id}")
            print("Type 'exit', 'quit', or Ctrl+C to end the session.")
            print("Type 'clear' to clear the current session.")
            print("-" * 50)
            
            await self.subscribe_to_events()
            
            # Main interaction loop
            while self.running:
                try:
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
                                        await self.chatbot.store_feedback(response, score)
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

async def run_cli(model, storage, interactive=True):
    """Run the command-line interface."""
    # Create the chatbot interface
    chatbot = ChatbotInterface(model, storage, interactive=interactive)
    
    # Create and start the CLI
    cli = CommandLineInterface(chatbot)
    try:
        await cli.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await cli.stop()
