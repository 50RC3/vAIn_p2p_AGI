import asyncio
import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

class ProgressIndicator:
    """Simple progress indicator for console UI."""
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.running = False
        self.task = None
    
    async def _animate(self):
        """Animate the progress indicator."""
        chars = ['-', '\\', '|', '/']
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {chars[i]} ")
            sys.stdout.flush()
            i = (i + 1) % len(chars)
            await asyncio.sleep(0.1)
        
        # Clear the indicator when done
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()
    
    async def start(self):
        """Start the progress indicator."""
        self.running = True
        self.task = asyncio.create_task(self._animate())
    
    async def stop(self):
        """Stop the progress indicator."""
        if self.running:
            self.running = False
            if self.task:
                await self.task

def format_response(response_text: str) -> str:
    """Format the response for display, handling empty responses."""
    if not response_text or response_text.strip() == "":
        return "[The model didn't generate any text. Please try a different question.]"
    return response_text

async def process_input(model: Any, storage: Any, message: str) -> Dict[str, Any]:
    """Process a single input message."""
    progress = ProgressIndicator("Generating response")
    await progress.start()
    try:
        from ai_core.chatbot.interface import ChatbotInterface
        interface = ChatbotInterface(model, storage)
        await interface.start_session()
        response = await interface.process_message(message)
        return {
            "text": response.text,
            "confidence": response.confidence,
            "response": response
        }
    finally:
        await progress.stop()

async def run_cli(model: Any, storage: Any, interactive: bool = True) -> None:
    """Run an interactive CLI session."""
    from ai_core.chatbot.interface import ChatbotInterface, ChatResponse
    
    # Initialize interface
    interface = ChatbotInterface(model, storage)
    session_id = await interface.start_session()
    logger.info(f"Started new session {session_id}")
    
    print(f"Started new chat session: {session_id}")
    print("Type 'exit', 'quit', or Ctrl+C to end the session.")
    print("Type 'clear' to clear the current session.")
    print("-" * 50)
    print()
    
    try:
        while True:
            # Get user input
            try:
                message = input("\nYou: ")
            except EOFError:
                break
                
            # Handle commands
            if message.lower() in ['exit', 'quit']:
                break
            elif message.lower() == 'clear':
                await interface.clear_session()
                session_id = await interface.start_session()
                print(f"Started new session: {session_id}")
                continue
            elif not message.strip():
                continue
                
            # Special direct handling for numeric inputs
            if message.strip().isdigit():
                logger.info(f"UI direct handling of digit: '{message}'")
                response = ChatResponse(
                    text=f"I see you entered the number {message.strip()}. How can I help you with this?",
                    confidence=1.0,
                    model_version="direct_handler",
                    latency=0.001
                )
                print(f"\nBot: {response.text}")
                
                # Still collect feedback
                if interactive:
                    try:
                        feedback = input("\nRate this response (0-1, or skip with Enter): ")
                        if feedback.strip():
                            score = float(feedback)
                            if 0 <= score <= 1:
                                await interface.store_feedback(response, score)
                                print(f"Feedback recorded: {score}")
                            else:
                                print("Invalid score. Please enter a number between 0 and 1.")
                    except ValueError:
                        print("Invalid input. Feedback skipped.")
                continue
                
            # Process regular messages
            try:
                # Show processing indicator
                progress = ProgressIndicator("Processing")
                await progress.start()
                
                try:
                    logger.info(f"Processing message: '{message}'")
                    response = await interface.process_message(message)
                    
                    # Format and display response
                    response_text = format_response(response.text)
                    print("\nBot:", response_text)
                finally:
                    await progress.stop()
                
                # Collect feedback if in interactive mode
                if interactive:
                    try:
                        feedback = input("\nRate this response (0-1, or skip with Enter): ")
                        if feedback.strip():
                            score = float(feedback)
                            if 0 <= score <= 1:
                                await interface.store_feedback(response, score)
                                print(f"Feedback recorded: {score}")
                            else:
                                print("Invalid score. Please enter a number between 0 and 1.")
                    except ValueError:
                        print("Invalid input. Feedback skipped.")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                print(f"\nError: {str(e)}")
                print("Try asking something else or type 'clear' to reset the session.")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        # Clean up
        if hasattr(interface, 'clear_session'):
            await interface.clear_session()
