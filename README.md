# Small LLM with Transformer Architecture

A complete implementation of a small language model using PyTorch Lightning with training, inference, and interactive web interface. This project demonstrates how to build, train, and deploy a transformer-based language model from scratch.

## Features

- **Transformer Architecture**: Custom implementation with sinusoidal positional encoding
- **Training Pipeline**: PyTorch Lightning with TensorBoard logging, early stopping, and checkpointing
- **Interactive Interface**: Gradio app with two generation modes:
  - Full response generation
  - Token-by-token prediction with probability visualization
- **Real-time Controls**: Interactive temperature adjustment with immediate probability updates
- **Configuration Management**: YAML-based configuration for easy hyperparameter tuning
- **Demo Script**: Complete testing suite to verify all components work correctly

## Project Structure

```
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer architecture
│   │   └── lightning_module.py # PyTorch Lightning module
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py        # Tokenization and vocabulary
│   │   └── dataset.py          # Dataset utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   └── generation.py       # Text generation utilities
│   └── app/
│       ├── __init__.py
│       └── gradio_app.py       # Gradio interface
├── train.py                    # Training script
├── demo.py                     # Demo script to test components
├── config.yaml                 # Configuration file
├── sample_corpus.txt           # Training data (user-provided)
├── requirements.txt            # Python dependencies
└── train_notebook.ipynb        # Jupyter notebook for experimentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd demo_llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Prepare your training data:**
   - Create a text file named `sample_corpus.txt` in the root directory
   - Add your training text data to this file
   - The file should contain the text you want your model to learn from

4. (Optional) Test the installation:
```bash
python demo.py
```

5. (Optional) Explore the Jupyter notebook for experimenting:
```bash
# Open in your IDE and select a kernel
train_notebook.ipynb
```

## Usage

### Quick Start

1. **Test the components** (recommended first step):
```bash
python demo.py
```

2. **Train a model**:
```bash
python train.py
```

3. **Run the interactive app**:
```bash
python -m src.app.gradio_app
```

### Data Preparation

Before training, you need to prepare your training data:

1. **Create `sample_corpus.txt`** in the root directory
2. **Add your text data** - any format works (paragraphs, sentences, etc.)
3. **Ensure sufficient data** - at least a few thousand words for meaningful training

**Example `sample_corpus.txt`:**
```
This is a paragraph of your training data. The model will learn from this text by processing it as a continuous sequence of words. You can add multiple paragraphs, stories, or any text format you want. The model uses a sliding window approach, so it will learn patterns from overlapping sequences of words.
```

### Training

The training script supports several options:

```bash
# Train with default configuration
python train.py

# Train with custom corpus
python train.py --corpus_path your_corpus.txt

# Train with custom config file
python train.py --config custom_config.yaml
```

**Monitor training progress:**
```bash
tensorboard --logdir=./logs
```

### Inference

**Run the Gradio app:**
```bash
# With default model (if available)
python -m src.app.gradio_app

# Or with the last trained model
python -m src.app.gradio_app --model_path ./checkpoints/transformer_lm-final.ckpt --vocab_path ./checkpoints/vocab.pkl

# Or with a specific version of the model (below with v01)
python -m src.app.gradio_app --model_path ./checkpoints/transformer_lm-best-v01.ckpt --vocab_path ./checkpoints/vocab-v01.pkl
```

The Gradio app provides:
- **Full Response Generation**: Generate complete text responses
- **Token-by-Token Prediction**: See next token probabilities in real-time
- **Interactive Controls**: Adjust temperature and other parameters
- **Probability Visualization**: Visual bars showing token probabilities

### Configuration

The project uses `config.yaml` for all hyperparameters and settings:

```yaml
hparams:
  vocab_size: 3000
  d_model: 32
  num_heads: 1
  num_layers: 2
  sequence_length: 32
  batch_size: 64
  learning_rate: 0.0001
  max_epochs: 100
  # ... more parameters

paths:
  corpus_path: "sample_corpus.txt"
  save_dir: "./checkpoints"
  log_dir: "./logs"
```

**Key configuration options:**
- **Model Architecture**: `d_model`, `num_heads`, `num_layers`, `d_ff`
- **Training**: `learning_rate`, `max_epochs`, `batch_size`, `sequence_length`
- **Data**: `vocab_size`, `train_split`, `val_split`
- **Paths**: `corpus_path`, `save_dir`, `log_dir`

### Demo Script

The `demo.py` script tests all components:

```bash
python demo.py
```

This will:
- Test tokenizer functionality
- Test dataset creation
- Test transformer model
- Test PyTorch Lightning integration
- Test text generation
- Provide next steps for training and deployment

### Jupyter Notebook

The `train_notebook.ipynb` provides an interactive environment for:

```bash
jupyter notebook train_notebook.ipynb
```

**Notebook features:**
- **Step-by-step training process** - Understand each component
- **Interactive debugging** - Inspect data, model, and training progress
- **Experimentation** - Try different hyperparameters

## What This Project Demonstrates

This project is a complete educational implementation of a small language model that shows:

- **Transformer Architecture**: How to build a transformer from scratch with attention mechanisms
- **Language Modeling**: Next-token prediction and text generation
- **PyTorch Lightning**: Modern training framework with callbacks, logging, and checkpointing
- **Text Processing**: Tokenization, vocabulary building, and dataset creation
- **Interactive Deployment**: Gradio-based web interface for model interaction
- **Configuration Management**: YAML-based hyperparameter management
- **Best Practices**: Proper project structure, testing, and documentation

## Expected Results

With the default configuration, you can expect:
- **Model Size**: ~100K parameters (very small, fast training)
- **Training Time**: Depends on corpus size, vocabulary, and the type of CPU/GPU used. below are estimates for standard PC with simple GPU:
  - **Small corpus** (~10K tokens): 15-30 minutes
  - **Medium corpus** (~100K tokens): 4-6 hours
  - **Large corpus** (~1M tokens): 15+ hours
- **Text Quality**: Coherent short phrases and sentences (limited by small model size)
- **Use Cases**: Educational purposes, understanding transformer internals, text generation experiments

**Training time factors:**
- **Corpus size** - More tokens = longer training
- **Vocabulary size** - Larger vocab = more parameters to learn
- **Sequence length** - Longer sequences = more computation per batch
- **Hardware** - GPU significantly faster than CPU

## Next Steps

After running the demo and training:
1. Experiment with different hyperparameters in `config.yaml`
2. Try different text corpora for domain-specific training
3. Increase model size for better text quality
4. Explore the Jupyter notebook for interactive experimentation
5. Modify the transformer architecture for your specific needs
