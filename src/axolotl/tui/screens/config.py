"""Configuration management screen for Axolotl TUI."""

import os
from pathlib import Path
from typing import Optional

import yaml
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Label,
    Log,
    Static,
    TextArea,
)

from axolotl.tui.screens.base import BaseScreen


class ConfigScreen(BaseScreen):
    """Configuration management screen."""

    BINDINGS = [
        Binding("ctrl+n", "new_config", "New Config"),
        Binding("ctrl+o", "open_config", "Open Config"),
        Binding("ctrl+s", "save_config", "Save Config"),
        Binding("ctrl+v", "validate_config", "Validate"),
        Binding("ctrl+e", "edit_mode", "Toggle Edit Mode"),
    ]

    CSS = """
    .config-container {
        layout: horizontal;
        height: 100%;
    }

    .file-browser {
        width: 30%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .config-editor {
        width: 70%;
        border: solid $secondary;
        padding: 1;
        margin: 1;
    }

    .config-form {
        height: 80%;
    }

    .config-actions {
        layout: horizontal;
        height: 3;
        align: center middle;
        padding: 1;
    }

    .config-actions Button {
        margin: 0 1;
    }

    TextArea {
        height: 100%;
    }

    .validation-log {
        height: 20%;
        border: solid $warning;
        padding: 1;
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
    """

    def __init__(self):
        """Initialize the config screen."""
        super().__init__(
            title="Configuration Management",
            subtitle="Create, edit, and validate Axolotl configurations",
        )
        self.current_config_path: Optional[Path] = None
        self.edit_mode = reactive(False)
        self.config_data = {}

    def compose(self) -> ComposeResult:
        """Compose the config screen layout."""
        yield Header()
        yield Container(
            Static("🦾 Configuration Management", classes="screen-title"),
            Static(
                "Create, edit, and validate Axolotl configurations",
                classes="screen-subtitle",
            ),
            Container(
                Container(
                    Label("Config Files"),
                    DirectoryTree(
                        (
                            Path("/workspace/configs")
                            if Path("/workspace/configs").exists()
                            else Path.cwd()
                        ),
                        id="config-tree",
                    ),
                    classes="file-browser",
                ),
                Container(
                    Container(
                        TextArea(
                            "",
                            language="yaml",
                            theme="monokai",
                            id="config-editor",
                            read_only=True,
                        ),
                        classes="config-form",
                    ),
                    Container(
                        Button("New", id="new-config", variant="primary"),
                        Button("Open", id="open-config", variant="primary"),
                        Button("Save", id="save-config", variant="success"),
                        Button("Validate", id="validate-config", variant="warning"),
                        Button("Edit Mode", id="toggle-edit", variant="default"),
                        Button("Load Example", id="load-example", variant="default"),
                        classes="config-actions",
                    ),
                    Container(
                        Log(id="validation-log"),
                        classes="validation-log",
                    ),
                    classes="config-editor",
                ),
                classes="config-container",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        tree = self.query_one("#config-tree", DirectoryTree)
        tree.show_root = False
        tree.guide_depth = 3

        log = self.query_one("#validation-log", Log)
        log.write_line("Ready to load configuration files...")

    @on(DirectoryTree.FileSelected)
    def handle_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection from the directory tree."""
        if event.path.suffix in [".yaml", ".yml"]:
            self.load_config_file(event.path)

    def load_config_file(self, path: Path) -> None:
        """Load a configuration file."""
        self.current_config_path = path
        try:
            with open(path, "r") as f:
                content = f.read()
                self.config_data = yaml.safe_load(content)

            editor = self.query_one("#config-editor", TextArea)
            editor.load_text(content)

            log = self.query_one("#validation-log", Log)
            log.clear()
            log.write_line(f"✅ Loaded: {path.name}")

        except Exception as e:
            log = self.query_one("#validation-log", Log)
            log.write_line(f"❌ Error loading {path.name}: {str(e)}")

    @on(Button.Pressed, "#new-config")
    def handle_new_config(self) -> None:
        """Create a new configuration."""
        template = """# Axolotl Configuration
base_model:
model_type:
tokenizer_type:

# Dataset Configuration
datasets:
  - path:
    type:

# Training Configuration
output_dir: ./outputs
num_epochs: 3
micro_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 0.00002
warmup_steps: 100
eval_steps: 100
save_steps: 500

# LoRA Configuration (optional)
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:

# Training optimizations
gradient_checkpointing: true
flash_attention: true
bf16: auto
tf32: true

# Logging
logging_steps: 10
wandb_project:
wandb_entity:
"""
        editor = self.query_one("#config-editor", TextArea)
        editor.load_text(template)
        editor.read_only = False
        self.edit_mode = True
        self.update_edit_button()

        log = self.query_one("#validation-log", Log)
        log.clear()
        log.write_line("📝 New configuration created. Edit and save when ready.")

    @on(Button.Pressed, "#save-config")
    def handle_save_config(self) -> None:
        """Save the current configuration."""
        editor = self.query_one("#config-editor", TextArea)
        content = editor.text

        if not content.strip():
            log = self.query_one("#validation-log", Log)
            log.write_line("⚠️ Cannot save empty configuration")
            return

        if not self.current_config_path:
            default_path = Path("/workspace/configs/new_config.yaml")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_config_path = default_path

        try:
            with open(self.current_config_path, "w") as f:
                f.write(content)

            log = self.query_one("#validation-log", Log)
            log.write_line(f"💾 Saved: {self.current_config_path.name}")
        except Exception as e:
            log = self.query_one("#validation-log", Log)
            log.write_line(f"❌ Error saving: {str(e)}")

    @on(Button.Pressed, "#validate-config")
    @work(thread=True)
    async def handle_validate_config(self) -> None:
        """Validate the current configuration."""
        editor = self.query_one("#config-editor", TextArea)
        content = editor.text

        if not content.strip():
            log = self.query_one("#validation-log", Log)
            log.write_line("⚠️ No configuration to validate")
            return

        log = self.query_one("#validation-log", Log)
        log.clear()
        log.write_line("🔍 Validating configuration...")

        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(content)
                temp_path = f.name

            from argparse import Namespace

            from axolotl.cli.config import check_user_config

            args = Namespace(
                config=temp_path,
                debug=False,
                debug_text_only=False,
                debug_num_examples=5,
                accelerate_config=None,
                multi_gpu=False,
            )

            check_user_config(args)

            log.write_line("✅ Configuration is valid!")

            os.unlink(temp_path)

        except Exception as e:
            log.write_line(f"❌ Validation failed: {str(e)}")
            if "temp_path" in locals():
                os.unlink(temp_path)

    @on(Button.Pressed, "#toggle-edit")
    def handle_toggle_edit(self) -> None:
        """Toggle edit mode for the configuration."""
        editor = self.query_one("#config-editor", TextArea)
        self.edit_mode = not self.edit_mode
        editor.read_only = not self.edit_mode
        self.update_edit_button()

        log = self.query_one("#validation-log", Log)
        if self.edit_mode:
            log.write_line("✏️ Edit mode enabled")
        else:
            log.write_line("👁️ View mode enabled")

    @on(Button.Pressed, "#load-example")
    async def handle_load_example(self) -> None:
        """Load an example configuration."""
        examples_dir = Path("/workspace/axolotl/examples")
        if not examples_dir.exists():
            log = self.query_one("#validation-log", Log)
            log.write_line("⚠️ Examples directory not found")
            return

        yaml_files = list(examples_dir.glob("**/*.yml")) + list(
            examples_dir.glob("**/*.yaml")
        )
        if yaml_files:
            self.load_config_file(yaml_files[0])
            log = self.query_one("#validation-log", Log)
            log.write_line(f"📚 Loaded example: {yaml_files[0].name}")

    def update_edit_button(self) -> None:
        """Update the edit button appearance."""
        button = self.query_one("#toggle-edit", Button)
        if self.edit_mode:
            button.variant = "warning"
            button.label = "Edit Mode: ON"
        else:
            button.variant = "default"
            button.label = "Edit Mode: OFF"

    def action_new_config(self) -> None:
        """Create a new configuration."""
        self.handle_new_config()

    def action_save_config(self) -> None:
        """Save the current configuration."""
        self.handle_save_config()

    def action_validate_config(self) -> None:
        """Validate the current configuration."""
        self.handle_validate_config()

    def action_edit_mode(self) -> None:
        """Toggle edit mode."""
        self.handle_toggle_edit()
