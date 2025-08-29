# Axolotl Multi-Task Golf Strategy Environment Setup

This document explains how to recreate the working environment on a new PC with GPU support.

## 🚀 Quick Setup Guide

### 1. Prerequisites
- **Python 3.11+** (confirmed working with Python 3.13)
- **NVIDIA GPU** with CUDA support
- **Git** for cloning the repository

### 2. Initial Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd axolotl

# Create virtual environment (recommended)
python -m venv axo-env
# Windows:
axo-env\Scripts\activate
# Linux/Mac:
source axo-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. GPU-Specific Setup
The environment will automatically detect and use GPU when available. Key improvements with GPU:
- **Training speed**: 4.5 min/step → ~30 seconds/step (estimated)
- **Memory**: Can handle larger batch sizes
- **Models**: Can train larger models or longer sequences

### 4. Verify Installation
```bash
# Test basic functionality
python train_lora_bethpage_strat.py --start quick --data-file "data/bethpage_black/train_multitask_case_fixed.jsonl"
```

## 📁 Project Architecture

### Core Training Infrastructure
- **`train_lora_bethpage_strat.py`**: Main training pipeline with multiple plans (quick, short, long, multitask)
- **`fix_and_infer_lora_v3.py`**: Inference pipeline with adapter loading and model merging
- **`check_inference_strategies.py`**: Validation and performance analysis

### Data Processing Pipeline
- **`create_better_multitask_dataset.py`**: Creates balanced multi-task training data
- **`fix_multitask_dataset.py`**: Standardizes completion formats
- **`fix_case_consistency.py`**: Fixes task label case inconsistencies
- **`process_hole_descriptions.py`**: Processes hole description data

### Key Datasets (Preserved)
- **`data/bethpage_black/train_multitask_case_fixed.jsonl`**: Latest corrected multi-task dataset (507 examples)
- **`data/bethpage_black/train_enhanced.jsonl`**: Single-task strategy dataset
- **`data/bethpage_black/basics.jsonl`**: Original training data

### Working Checkpoints
Will be recreated during training. Key checkpoint types:
- **quick_test**: 7-step validation training (~30 minutes)
- **short**: 50-step training (~4 hours)
- **multitask**: Full multi-task training (8+ hours)

## 🔧 Environment Variables & Configuration

### Training Plans Available
```python
PLAN_CONFIGS = {
    "quick": {"num_epochs": 7, "description": "Quick validation test"},
    "short": {"num_epochs": 50, "description": "Medium training run"},
    "long": {"num_epochs": 200, "description": "Full training run"},
    "multitask": {"num_epochs": 107, "description": "Multi-task optimized"}
}
```

### Key Configuration Files
- **`requirements.txt`**: Core dependencies
- **`pyproject.toml`**: Project metadata
- **`.gitignore`**: Excludes large outputs and checkpoints

## 🎯 Current Project Status

### Achievements
✅ **Single-task strategy model**: 94.5% accuracy on strategy selection  
✅ **Multi-task dataset creation**: Balanced 507-example dataset with task prefixing  
✅ **Training pipeline optimization**: Realistic timing estimates and checkpoint management  
✅ **Data quality fixes**: Format standardization and case consistency corrections  

### Current Challenge
❌ **Multi-task learning**: Despite good loss convergence, task recognition fails completely (0% strategy task success)

### Next Steps for GPU Environment
1. **Validate faster training times** with GPU acceleration
2. **Try alternative multi-task architectures** with improved compute capacity
3. **Consider larger models** that might handle multi-task learning better
4. **Experiment with different task prefixing approaches**

## 🔍 Key Scripts Usage

### Training
```bash
# Quick validation (7 steps, ~30 min on GPU)
python train_lora_bethpage_strat.py --start quick --data-file "data/bethpage_black/train_multitask_case_fixed.jsonl"

# Full multi-task training
python train_lora_bethpage_strat.py --start multitask --data-file "data/bethpage_black/train_multitask_case_fixed.jsonl"
```

### Inference & Validation
```bash
# Run inference
python fix_and_infer_lora_v3.py --adapter_dir "outputs/bethpage-lora/checkpoint-quick" --jsonl "data/bethpage_black/train_multitask_case_fixed.jsonl" --output "outputs/bethpage-lora/test_results.jsonl"

# Validate results
python check_inference_strategies.py --inference-file "outputs/bethpage-lora/test_results.jsonl" --data-file "data/bethpage_black/train_multitask_case_fixed.jsonl" --log-file "outputs/bethpage-lora/validation_log.txt"
```

### Data Processing
```bash
# Create multi-task dataset
python create_better_multitask_dataset.py

# Fix format inconsistencies
python fix_multitask_dataset.py

# Fix case consistency
python fix_case_consistency.py
```

## 🐛 Known Issues & Solutions

### Issue: Multi-task Learning Failure
- **Problem**: 0% success rate on strategy tasks despite task prefixing
- **Attempted Fixes**: Format standardization, case consistency, balanced datasets
- **Status**: Requires architectural investigation on GPU environment

### Issue: Training Time Estimation
- **Problem**: Original estimates were too optimistic
- **Solution**: Calibrated to 4.5 min/step on CPU (should be ~30s/step on GPU)

### Issue: Dataset Quality
- **Problem**: Mixed completion formats, case inconsistencies
- **Solution**: Comprehensive data cleaning pipeline implemented

## 📊 Performance Metrics

### Single-Task Model (Baseline)
- **Strategy Selection**: 94.5% accuracy
- **Training Time**: ~8 hours for full training
- **Model Size**: LoRA fine-tuned

### Multi-Task Model (Current Challenge)
- **Strategy Selection**: 0% accuracy (complete failure)
- **Description Synthesis**: Not properly evaluated due to strategy failures
- **Loss Convergence**: Good (4.8275 → 3.8536 over 7 steps)
- **Root Cause**: Task recognition failure, not loss optimization

## 🚀 GPU Acceleration Benefits

### Expected Improvements
- **Training Speed**: 10-15x faster training
- **Batch Size**: Larger batches for better gradient estimates
- **Model Capacity**: Ability to train larger models
- **Experimentation**: Faster iteration cycles for debugging multi-task issues

### GPU-Specific Considerations
- **CUDA Setup**: Will be handled automatically by PyTorch/Transformers
- **Memory Management**: Monitor GPU memory usage during training
- **Batch Size Tuning**: Can increase batch sizes with more VRAM

---

*This environment preserves all working scripts, datasets, and configurations needed to continue multi-task learning research with GPU acceleration.*
