# 📓 Jupyter Notebooks

This directory contains Jupyter notebooks for exploring and using the LLM fine-tuning framework.

## Available Notebooks

### 01_quick_start.ipynb
A comprehensive quick start guide that demonstrates:
- Setting up the environment
- Creating sample data
- Loading and modifying configurations
- Preprocessing data
- Training a model (optional)
- Evaluating results

### 02_data_exploration.ipynb
Explores different data formats and preprocessing techniques:
- Loading various data formats (JSONL, JSON, CSV, TXT)
- Data validation and cleaning
- Custom instruction templates
- Data visualization and statistics

### 03_model_comparison.ipynb
Compares different models and configurations:
- Training multiple models with different settings
- Performance comparison
- Resource usage analysis
- Best practices recommendations

### 04_advanced_training.ipynb
Advanced training techniques:
- Custom training loops
- Hyperparameter optimization
- Multi-GPU training
- Gradient checkpointing

### 05_evaluation_analysis.ipynb
Comprehensive evaluation and analysis:
- Multiple evaluation metrics
- Error analysis
- Model interpretability
- Performance visualization

## Running Notebooks

1. **Install Jupyter**: `pip install jupyter`
2. **Start Jupyter**: `jupyter notebook`
3. **Navigate**: Open the notebook you want to run
4. **Run cells**: Execute cells sequentially

## Requirements

Make sure you have installed the project dependencies:
```bash
pip install -r requirements.txt
```

## Tips

- Start with `01_quick_start.ipynb` to understand the basics
- Modify configurations in the notebooks to experiment with different settings
- Use GPU acceleration for faster training (if available)
- Save your trained models and results for later analysis

## Contributing

Feel free to add new notebooks or improve existing ones:
- Add clear documentation and comments
- Include example outputs where helpful
- Test notebooks with different configurations
- Share interesting findings and experiments