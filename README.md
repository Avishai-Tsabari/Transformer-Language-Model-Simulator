# Small LLM with Transformer Architecture

A complete implementation of a small language model using PyTorch Lightning with training, inference, and interactive web interface.

## Features

- **Transformer Architecture**: Custom implementation with sinusoidal positional encoding
- **Training Pipeline**: PyTorch Lightning with TensorBoard logging, early stopping, and checkpointing
- **Interactive Interface**: Gradio app with two generation modes:
  - Full response generation
  - Token-by-token prediction with probability visualization
- **Real-time Controls**: Interactive temperature adjustment with immediate probability updates

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
├── sample_corpus.txt           # Example training data
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Prepare your text corpus in `sample_corpus.txt`
2. Run training:
```bash
python train.py
```

3. Monitor training with TensorBoard:
```bash
tensorboard --logdir=./logs
```

### Inference

Run the Gradio app:
```bash
python -m src.app.gradio_app
```

## Configuration

Modify hyperparameters in `train.py` or create a config file for different model sizes and training settings.
