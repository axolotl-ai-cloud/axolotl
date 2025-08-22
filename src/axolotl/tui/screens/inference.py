"""Inference and testing screen for Axolotl TUI."""

from pathlib import Path
from typing import Dict, List, Optional

from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import (
    Button,
    Input,
    Label,
    Log,
    Select,
    Static,
    TextArea,
)

from axolotl.tui.screens.base import BaseScreen


class InferenceScreen(BaseScreen):
    """Inference and testing screen."""

    BINDINGS = [
        Binding("ctrl+enter", "send_message", "Send"),
        Binding("ctrl+c", "clear_chat", "Clear"),
        Binding("ctrl+l", "load_model", "Load Model"),
        Binding("ctrl+s", "save_chat", "Save Chat"),
    ]

    CSS = """
    .inference-container {
        layout: horizontal;
        height: 100%;
    }

    .model-selector {
        width: 30%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .chat-interface {
        width: 70%;
        border: solid $secondary;
        padding: 1;
        margin: 1;
    }

    .chat-history {
        height: 70%;
        border: solid $primary;
        padding: 1;
        margin: 0 0 1 0;
    }

    .input-area {
        height: 20%;
        border: solid $warning;
        padding: 1;
        margin: 0 0 1 0;
    }

    .chat-controls {
        layout: horizontal;
        height: 4;
        align: center middle;
        padding: 1;
    }

    .chat-controls Button {
        margin: 0 1;
    }

    .model-info {
        padding: 1;
        border: solid $surface;
        margin: 1 0;
    }

    .screen-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    .screen-subtitle {
        text-align: center;
        padding: 0 0 1 0;
        color: $text-muted;
    }

    TextArea {
        height: 100%;
    }

    Log {
        height: 100%;
    }
    """

    def __init__(self):
        """Initialize the inference screen."""
        super().__init__(
            title="Inference & Testing", subtitle="Interactive chat and model testing"
        )
        self.loaded_model: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []

    def compose(self) -> ComposeResult:
        """Compose the inference screen layout."""
        yield Container(
            Static("🦾 Inference & Testing", classes="screen-title"),
            Static("Interactive chat and model testing", classes="screen-subtitle"),
            Container(
                Container(
                    Label("Model Selection"),
                    Select(
                        [("No model loaded", "none")],
                        id="model-select",
                        value="none",
                    ),
                    Container(
                        Button("Load Model", id="load-model", variant="primary"),
                        Button("Unload", id="unload-model", variant="default"),
                        Button("Gradio UI", id="gradio-ui", variant="success"),
                    ),
                    Container(
                        Static("No model loaded", id="model-status"),
                        classes="model-info",
                    ),
                    Label("Inference Parameters"),
                    Container(
                        Label("Temperature:"),
                        Input(value="0.7", id="temperature"),
                        Label("Max Tokens:"),
                        Input(value="256", id="max-tokens"),
                        Label("Top P:"),
                        Input(value="0.9", id="top-p"),
                    ),
                    classes="model-selector",
                ),
                Container(
                    Container(
                        Log(id="chat-history"),
                        classes="chat-history",
                    ),
                    Container(
                        TextArea(
                            id="message-input",
                        ),
                        classes="input-area",
                    ),
                    Container(
                        Button("Send [Ctrl+Enter]", id="send", variant="primary"),
                        Button("Clear Chat", id="clear", variant="warning"),
                        Button("Save Chat", id="save-chat", variant="default"),
                        Button("Load Examples", id="load-examples", variant="default"),
                        classes="chat-controls",
                    ),
                    classes="chat-interface",
                ),
                classes="inference-container",
            ),
            id="content",
        )

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.load_available_models()

        chat = self.query_one("#chat-history", Log)
        chat.write_line("💬 Welcome to Axolotl Inference!")
        chat.write_line("Load a model to start chatting.")

    @work(thread=True)
    async def load_available_models(self) -> None:
        """Load list of available models."""
        models = [("No model loaded", "none")]

        chat = self.query_one("#chat-history", Log)
        chat.write_line("🔍 Scanning for available models...")

        # Check for trained models
        outputs_dir = Path("./outputs")
        chat.write_line(f"Checking outputs directory: {outputs_dir.absolute()}")
        if outputs_dir.exists():
            found_models = 0
            for model_dir in outputs_dir.glob("*"):
                if model_dir.is_dir():
                    # Look for various model file types
                    model_files = (
                        list(model_dir.glob("pytorch_model.bin"))
                        + list(model_dir.glob("model.safetensors"))
                        + list(model_dir.glob("*.bin"))
                        + list(model_dir.glob("*.safetensors"))
                    )
                    if model_files:
                        models.append((model_dir.name, str(model_dir)))
                        found_models += 1
            chat.write_line(f"Found {found_models} trained models in outputs/")
        else:
            chat.write_line("outputs/ directory not found")

        # Add some example/demo models for testing
        models.extend(
            [
                ("Demo: GPT-2 Small", "gpt2"),
                ("Demo: TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                ("Demo: Phi-2", "microsoft/phi-2"),
            ]
        )

        select = self.query_one("#model-select", Select)
        select.set_options(models)
        chat.write_line(f"✅ Loaded {len(models)} models in dropdown")

    @on(Button.Pressed, "#load-model")
    @work(thread=True)
    async def handle_load_model(self) -> None:
        """Load selected model for inference."""
        select = self.query_one("#model-select", Select)
        if select.value == "none":
            return

        chat = self.query_one("#chat-history", Log)
        chat.write_line(f"🔄 Loading model: {select.value}")

        status = self.query_one("#model-status", Static)
        status.update("Loading...")

        try:
            # Simulate model loading (in real implementation, would load the actual model)
            import time

            time.sleep(2)  # Simulate loading time

            self.loaded_model = select.value
            status.update(f"✅ Loaded: {Path(select.value).name}")
            chat.write_line("✅ Model loaded successfully!")
            chat.write_line("You can now start chatting.")

        except Exception as e:
            status.update("❌ Failed to load")
            chat.write_line(f"❌ Failed to load model: {str(e)}")

    @on(Button.Pressed, "#send")
    async def handle_send_message(self) -> None:
        """Send message to model."""
        if not self.loaded_model:
            chat = self.query_one("#chat-history", Log)
            chat.write_line("⚠️ Please load a model first")
            return

        message_input = self.query_one("#message-input", TextArea)
        message = message_input.text.strip()

        if not message:
            return

        # Add user message to chat
        chat = self.query_one("#chat-history", Log)
        chat.write_line(f"👤 User: {message}")

        # Clear input
        message_input.clear()

        # Add to history
        self.chat_history.append({"role": "user", "content": message})

        # Generate response (placeholder)
        self.generate_response(message)

    @on(TextArea.Changed, "#message-input")
    def on_message_input_changed(self, event: TextArea.Changed) -> None:
        """Handle changes to the message input."""
        # This could be used for features like typing indicators
        pass

    def on_key(self, event: events.Key) -> None:
        """Handle key events globally."""
        # Check if we're focused on the message input and Ctrl+Enter is pressed
        focused = self.focused
        if focused and focused.id == "message-input" and event.key == "ctrl+enter":
            event.prevent_default()
            self.handle_send_message()

    @work(thread=True)
    async def generate_response(self, message: str) -> None:
        """Generate model response."""
        chat = self.query_one("#chat-history", Log)
        chat.write_line("🤖 Assistant: Thinking...")

        try:
            # Get inference parameters
            float(self.query_one("#temperature", Input).value)
            int(self.query_one("#max-tokens", Input).value)
            float(self.query_one("#top-p", Input).value)

            if not self.loaded_model or self.loaded_model == "none":
                response = "I don't have a model loaded yet. Please load a model first using the 'Load Model' button."
            elif self.loaded_model.startswith("gpt2"):
                # Simple response for GPT-2
                responses = [
                    f"Thanks for your message: '{message}'. I'm a GPT-2 model running in demo mode.",
                    "I understand you're testing the interface. GPT-2 models are great for experimentation!",
                    "This is a simulated GPT-2 response. In a real setup, I'd generate text based on your input.",
                    f"GPT-2 here! You said: '{message}'. I'd normally continue this conversation creatively.",
                ]
                import random

                response = random.choice(responses)
            elif "llama" in self.loaded_model.lower():
                # Response for Llama models
                response = f"🦙 LLaMA model here! You asked: '{message}'. I'm designed for helpful, harmless, and honest conversations. How can I assist you today?"
            elif "phi" in self.loaded_model.lower():
                # Response for Phi models
                response = f"Phi model responding! Your message: '{message}'. I'm optimized for reasoning and code tasks. What would you like to explore?"
            else:
                # Generic response for other models
                response = f"Model '{self.loaded_model}' responding to: '{message}'. I'm ready to help with your questions!"

            # Simulate inference time
            import time

            time.sleep(0.5)

            # Clear the "thinking" message and show response
            chat.write_line(f"🤖 Assistant: {response}")

            # Add to history
            self.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            chat.write_line(f"❌ Error generating response: {str(e)}")

    @on(Button.Pressed, "#clear")
    def handle_clear_chat(self) -> None:
        """Clear chat history."""
        chat = self.query_one("#chat-history", Log)
        chat.clear()
        self.chat_history = []
        chat.write_line("💬 Chat cleared. Start a new conversation!")

    @on(Button.Pressed, "#save-chat")
    def handle_save_chat(self) -> None:
        """Save chat history to file."""
        if not self.chat_history:
            chat = self.query_one("#chat-history", Log)
            chat.write_line("⚠️ No chat history to save")
            return

        try:
            import json
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(self.chat_history, f, indent=2)

            chat = self.query_one("#chat-history", Log)
            chat.write_line(f"💾 Chat saved to {filename}")

        except Exception as e:
            chat = self.query_one("#chat-history", Log)
            chat.write_line(f"❌ Error saving chat: {str(e)}")

    @on(Button.Pressed, "#load-examples")
    def handle_load_examples(self) -> None:
        """Load example prompts."""
        examples = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of fine-tuning language models?",
            "Describe the difference between supervised and unsupervised learning.",
        ]

        chat = self.query_one("#chat-history", Log)
        chat.write_line("📚 Example prompts:")
        for i, example in enumerate(examples, 1):
            chat.write_line(f"{i}. {example}")
        chat.write_line("Copy and paste any example to try it out!")

    @on(Button.Pressed, "#gradio-ui")
    @work(thread=True)
    async def handle_gradio_ui(self) -> None:
        """Launch Gradio web interface."""
        chat = self.query_one("#chat-history", Log)
        chat.write_line("🌐 Launching Gradio web interface...")

        try:
            import subprocess

            if self.loaded_model:
                cmd = [
                    "python",
                    "-m",
                    "axolotl.cli.inference",
                    self.loaded_model,
                    "--gradio",
                ]
            else:
                chat.write_line("⚠️ No model loaded. Loading default interface...")
                cmd = ["python", "-m", "axolotl.cli.inference", "--gradio"]

            subprocess.Popen(cmd)
            chat.write_line("✅ Gradio interface launched! Check your browser.")

        except Exception as e:
            chat.write_line(f"❌ Error launching Gradio: {str(e)}")

    @on(Button.Pressed, "#unload-model")
    def handle_unload_model(self) -> None:
        """Unload current model."""
        self.loaded_model = None
        status = self.query_one("#model-status", Static)
        status.update("No model loaded")

        select = self.query_one("#model-select", Select)
        select.value = "none"

        chat = self.query_one("#chat-history", Log)
        chat.write_line("🔄 Model unloaded")

    def action_send_message(self) -> None:
        """Send message action."""
        self.handle_send_message()

    def action_clear_chat(self) -> None:
        """Clear chat action."""
        self.handle_clear_chat()

    def action_load_model(self) -> None:
        """Load model action."""
        self.handle_load_model()

    def action_save_chat(self) -> None:
        """Save chat action."""
        self.handle_save_chat()
