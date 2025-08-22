# Axolotl TUI (Terminal User Interface)

A comprehensive Terminal User Interface for Axolotl, providing an interactive way to manage configurations, training jobs, datasets, models, and system monitoring.

## Features

### 🏠 Main Dashboard
- **Welcome Screen**: Central hub with quick access to all features
- **Keyboard Navigation**: Efficient navigation with keyboard shortcuts
- **Screen Management**: Easy switching between different functional areas

### 📝 Configuration Management
- **YAML Editor**: Syntax-highlighted editor for Axolotl configurations
- **Real-time Validation**: Instant config validation with detailed error reporting
- **File Browser**: Navigate and select configuration files
- **Template Loading**: Load example configurations
- **Remote Config Support**: Load configurations from URLs

**Key Shortcuts:**
- `Ctrl+N`: New configuration
- `Ctrl+S`: Save configuration
- `Ctrl+V`: Validate configuration
- `Ctrl+E`: Toggle edit mode

### 🚀 Training Management
- **Job Launcher**: Start training with different launchers (accelerate, torchrun)
- **Real-time Monitoring**: Live training progress and metrics
- **Loss Visualization**: Sparkline charts for loss curves
- **Job Control**: Start, stop, resume, and manage multiple training jobs
- **Log Streaming**: Real-time log viewing and filtering

**Key Shortcuts:**
- `Ctrl+T`: New training job
- `Ctrl+R`: Resume training
- `Ctrl+X`: Stop training
- `R`: Refresh status

### 📊 Dataset Management
- **Dataset Browser**: Explore local and remote datasets
- **Preview & Statistics**: View dataset samples and metadata
- **Preprocessing**: Run dataset preprocessing with progress tracking
- **HuggingFace Integration**: Download and manage HF datasets
- **Format Detection**: Automatic dataset format recognition

**Key Shortcuts:**
- `Ctrl+P`: Preprocess dataset
- `Ctrl+V`: Preview dataset
- `Ctrl+I`: Dataset information
- `R`: Refresh dataset list

### 🤖 Model Management
- **Model Discovery**: Automatically find trained models
- **LoRA Operations**: Merge LoRA adapters with base models
- **Quantization**: Quantize models for deployment
- **Evaluation**: Run model evaluation benchmarks
- **Storage Info**: View model sizes and storage details

**Key Shortcuts:**
- `Ctrl+M`: Merge LoRA
- `Ctrl+Q`: Quantize model
- `Ctrl+E`: Evaluate model
- `R`: Refresh model list

### 💬 Inference & Testing
- **Interactive Chat**: Chat interface for model testing
- **Parameter Tuning**: Adjust inference parameters (temperature, top-p, max tokens)
- **Model Loading**: Load and switch between different models
- **Chat History**: Save and load conversation history
- **Gradio Integration**: Launch Gradio web interface

**Key Shortcuts:**
- `Ctrl+Enter`: Send message
- `Ctrl+C`: Clear chat
- `Ctrl+L`: Load model
- `Ctrl+S`: Save chat

### 📈 System Monitoring
- **Resource Monitoring**: Real-time CPU, GPU, and memory usage
- **Process Management**: View and manage running processes
- **Performance Graphs**: Historical usage charts with sparklines
- **GPU Information**: Detailed GPU status and memory usage
- **Temperature Monitoring**: System temperature tracking

**Key Shortcuts:**
- `R`: Refresh metrics
- `Ctrl+K`: Kill selected process

## Installation

### Dependencies
```bash
pip install textual==1.0.0 rich==14.1.0
```

### Launch TUI
```bash
# From command line
python -m axolotl.cli.main tui

# From Python code
from axolotl.tui.app import run
run()
```

## Architecture

### Screen Structure
```
AxolotlTUI (Main App)
├── WelcomeScreen (Dashboard)
├── ConfigScreen (Configuration Management)
├── TrainingScreen (Training Management)
├── DatasetScreen (Dataset Management)
├── ModelScreen (Model Management)
├── InferenceScreen (Inference & Testing)
└── MonitorScreen (System Monitoring)
```

### Key Components
- **BaseScreen**: Common functionality for all screens
- **Screen Navigation**: Stack-based screen management
- **Event Handling**: Reactive UI updates
- **Background Tasks**: Non-blocking operations
- **State Management**: Shared application state

### Integration Points
- **CLI Commands**: Seamless integration with existing axolotl CLI
- **Configuration System**: Uses axolotl's native config loading
- **Training Pipeline**: Integrates with axolotl training functions
- **Model Loading**: Compatible with axolotl model management

## Usage Examples

### 1. Creating a New Configuration
1. Launch TUI: `python -m axolotl.cli.main tui`
2. Select "Configuration Management" or press `C`
3. Press `Ctrl+N` for new configuration
4. Edit the template configuration
5. Press `Ctrl+V` to validate
6. Press `Ctrl+S` to save

### 2. Starting a Training Job
1. Navigate to "Training Management" or press `T`
2. Press `Ctrl+T` for new training job
3. Select configuration file and launcher
4. Monitor progress in real-time
5. View loss curves and logs

### 3. Interactive Model Testing
1. Go to "Inference & Testing" or press `I`
2. Load a trained model with `Ctrl+L`
3. Adjust inference parameters as needed
4. Start chatting with the model
5. Save conversation with `Ctrl+S`

## Navigation

### Global Shortcuts
- `Ctrl+Q`: Quit application
- `Escape`: Go back/close current screen
- `Tab`: Navigate between UI elements
- `Enter`: Select/activate element
- `Space`: Toggle switches/checkboxes

### Screen Shortcuts
Each screen has specific shortcuts displayed in the footer. Common patterns:
- `Ctrl+[Letter]`: Primary actions
- `R`: Refresh/reload
- `F1-F12`: Function keys for advanced features

## Customization

### Themes
The TUI uses Textual's theming system and can be customized by modifying the CSS in each screen class.

### Adding New Screens
1. Create a new screen class inheriting from `BaseScreen`
2. Implement the `compose()` method for UI layout
3. Add event handlers for user interactions
4. Register the screen in the main app navigation

### Extending Functionality
- Add new widgets to existing screens
- Implement custom data visualization
- Integrate with external tools and APIs
- Add new keyboard shortcuts

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure textual and rich are installed
2. **Permission Errors**: Check file system permissions for config directories
3. **GPU Monitoring**: Install pynvml for GPU monitoring features
4. **Config Validation**: Ensure axolotl dependencies are properly installed

### Debug Mode
Launch with debug logging:
```bash
TEXTUAL_LOG=DEBUG python -m axolotl.cli.main tui
```

### Performance
- Use `Ctrl+\` to open Textual's debug console
- Monitor memory usage with the system monitor
- Disable auto-refresh for better performance on slower systems

## Contributing

The TUI is designed to be extensible. Contributions are welcome for:
- New screen implementations
- Enhanced visualizations
- Better keyboard navigation
- Additional integrations
- Performance improvements

See the main Axolotl repository for contribution guidelines.
