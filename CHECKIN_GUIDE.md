# Files to Check Into Git Repository

## 🔧 Core Infrastructure (MUST CHECK IN)

### Training Pipeline
- `train_lora_bethpage_strat.py` - Main training pipeline with multiple plans
- `fix_and_infer_lora_v3.py` - Inference pipeline with adapter loading
- `check_inference_strategies.py` - Validation and performance analysis

### Data Processing Scripts
- `create_better_multitask_dataset.py` - Creates balanced multi-task data
- `fix_multitask_dataset.py` - Standardizes completion formats  
- `fix_case_consistency.py` - Fixes task label case inconsistencies
- `process_hole_descriptions.py` - Processes hole description data

### Essential Datasets
- `data/bethpage_black/train_multitask_case_fixed.jsonl` - Latest corrected dataset (507 examples)
- `data/bethpage_black/train_enhanced.jsonl` - Single-task strategy dataset
- `data/bethpage_black/basics.jsonl` - Original training data
- `data/bethpage_black/train.jsonl` - Base training data
- `data/bethpage_black/val.jsonl` - Validation data

### Configuration Files
- `requirements.txt` - Core dependencies (CRITICAL)
- `pyproject.toml` - Project metadata
- `.gitignore` - Exclude large files (already configured)
- `ENVIRONMENT_SETUP.md` - Setup documentation

## 📊 Analysis & Validation Scripts (USEFUL TO PRESERVE)

### Performance Analysis
- `validate_enhanced_model.py` - Model validation utilities
- `manual_verify.py` - Manual verification tools
- `runner.py` - Execution runner utilities

### Data Analysis
- `scripts/analyze_validation_mismatches.py` - Mismatch analysis
- `scripts/simple_mismatch_analysis.py` - Simple analysis tools
- `scripts/create_strategy_logic_dataset.py` - Dataset creation utilities

## 🗑️ Files You Can SKIP (Throwaway/Generated)

### Virtual Environments (NEVER CHECK IN)
- `axo-env/` - Virtual environment (recreate on new PC)
- `axo-env-slim/` - Alternate virtual environment
- `lora-env/` - LoRA-specific environment

### Generated Outputs (ALREADY GITIGNORED)
- `outputs/` - All training outputs and checkpoints
- `model-out/` - Model outputs
- `last_run_prepared/` - Temporary processing files
- `__pycache__/` - Python cache files

### Experimental/Debug Scripts (OPTIONAL)
- `debug_lora*.py` - Debug scripts (throwaway)
- `toy_*.py` - Toy/test scripts
- `verify*.py` - One-off verification scripts
- `inference*.py` - Legacy inference scripts
- `train_lora.py` - Basic training (superseded)

### Generated Data Files (LARGE)
- `bethpage_holes.jsonl` - Can be regenerated
- `hole_descriptions_input.json` - Source data file
- Any `.log` files - Training logs

## 🎯 Git Commands for Check-in

```bash
# Add core infrastructure
git add train_lora_bethpage_strat.py
git add fix_and_infer_lora_v3.py 
git add check_inference_strategies.py

# Add data processing pipeline
git add create_better_multitask_dataset.py
git add fix_multitask_dataset.py
git add fix_case_consistency.py
git add process_hole_descriptions.py

# Add essential datasets
git add data/bethpage_black/train_multitask_case_fixed.jsonl
git add data/bethpage_black/train_enhanced.jsonl
git add data/bethpage_black/basics.jsonl
git add data/bethpage_black/train.jsonl
git add data/bethpage_black/val.jsonl

# Add configuration
git add requirements.txt
git add pyproject.toml
git add ENVIRONMENT_SETUP.md
git add .gitignore

# Add useful analysis scripts
git add validate_enhanced_model.py
git add scripts/

# Commit everything
git commit -m "Complete multi-task learning environment with GPU transition support

- Core training pipeline with multiple training plans
- Data processing and quality fixes (case consistency, format standardization)  
- Latest corrected multi-task dataset (507 balanced examples)
- Inference and validation infrastructure
- Comprehensive setup documentation for GPU environment
- Analysis tools for debugging multi-task learning issues

Ready for GPU acceleration and continued multi-task research."

# Push to remote
git push origin main
```

## 📋 Pre-Migration Checklist

- [ ] **Core Scripts**: All essential .py files added
- [ ] **Datasets**: Critical .jsonl files included  
- [ ] **Dependencies**: requirements.txt is complete and current
- [ ] **Documentation**: ENVIRONMENT_SETUP.md explains GPU transition
- [ ] **Gitignore**: Excludes large generated files and environments
- [ ] **Working State**: Latest case-fixed dataset represents current progress

## 🚀 New PC Setup Verification

After cloning on new PC with GPU:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Quick test**: Run `python train_lora_bethpage_strat.py --start quick`
3. **GPU verification**: Check that CUDA is detected and training is faster
4. **Continue research**: Pick up multi-task learning investigation with better compute

This checkin preserves the complete working environment while excluding unnecessary files.
