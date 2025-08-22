"""Inference and testing screen for Axolotl TUI."""

from pathlib import Path
from typing import Dict, List, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Input,
    Label,
    Log,
    Pretty,
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
        border: solid $info;
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
                        [("none", "No model loaded")],
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
                        Log(id="chat-history", wrap=True, highlight=True),
                        classes="chat-history",
                    ),
                    Container(
                        TextArea(
                            placeholder="Type your message here...",
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
        models = [("none", "No model loaded")]

        # Check for trained models
        outputs_dir = Path("./outputs")
        if outputs_dir.exists():
            for model_dir in outputs_dir.glob("*"):
                if model_dir.is_dir() and (model_dir / "pytorch_model.bin").exists():
                    models.append((str(model_dir), model_dir.name))

        # Check for HuggingFace models in cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "transformers"
        if hf_cache.exists():
            for model_dir in hf_cache.glob("models--*"):
                if model_dir.is_dir():
                    model_name = model_dir.name.replace("models--", "").replace(
                        "--", "/"
                    )
                    models.append((str(model_dir), f"HF: {model_name}"))

        select = self.query_one("#model-select", Select)
        select.set_options(models)

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
        await self.generate_response(message)

    @work(thread=True)
    async def generate_response(self, message: str) -> None:
        """Generate model response (placeholder implementation)."""
        chat = self.query_one("#chat-history", Log)
        chat.write_line("🤖 Assistant: Thinking...")

        try:
            # Get inference parameters
            temperature = float(self.query_one("#temperature", Input).value)
            max_tokens = int(self.query_one("#max-tokens", Input).value)
            top_p = float(self.query_one("#top-p", Input).value)

            # Placeholder response (in real implementation, would call the model)
            import time

            time.sleep(1)  # Simulate inference time

            response = f"This is a placeholder response to: '{message}'. In a real implementation, this would be generated by the loaded model using the parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}."

            # Update chat with response
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
