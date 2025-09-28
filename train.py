"""
Training script for the transformer language model.
"""
import os
import argparse
import torch
import pytorch_lightning as pl
import yaml
import shutil
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from src.data.tokenizer import load_text_corpus
from src.data.tokenizer_factory import TokenizerFactory, create_sample_corpus
from src.data.dataset import TextDataModule
from src.model.lightning_module import TransformerLightningModule


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to YAML configuration file")
    parser.add_argument("--corpus_path", type=str, default=None, 
                       help="Override corpus path from config")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample corpus if it doesn't exist")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        print("Please ensure config.yaml exists or specify a different config file with --config")
        return
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract configuration sections
    hparams = config["hparams"]
    paths = config["paths"]
    general = config["general"]
    
    # Override corpus path if provided via command line
    if args.corpus_path:
        paths["corpus_path"] = args.corpus_path
    
    # Extract hyperparameters (matching notebook approach)
    vocab_size = hparams["vocab_size"]
    d_model = hparams["d_model"]
    num_heads = hparams["num_heads"]
    num_layers = hparams["num_layers"]
    d_ff = hparams["d_ff"]
    sequence_length = hparams["sequence_length"]
    batch_size = hparams["batch_size"]
    learning_rate = hparams["learning_rate"]
    max_epochs = hparams["max_epochs"]
    patience = hparams["patience"]
    min_delta = hparams["min_delta"]
    warmup_steps = hparams["warmup_steps"]
    weight_decay = hparams["weight_decay"]
    dropout = hparams["dropout"]
    train_split = hparams["train_split"]
    val_split = hparams["val_split"]
    num_workers = hparams["num_workers"]
    accelerator = hparams["accelerator"]
    devices = hparams["devices"]
    precision = hparams["precision"]
    gradient_clip_val = hparams["gradient_clip_val"]
    accumulate_grad_batches = hparams["accumulate_grad_batches"]
    log_every_n_steps = hparams["log_every_n_steps"]
    val_check_interval = hparams["val_check_interval"]
    save_top_k = hparams["save_top_k"]
    monitor = hparams["monitor"]
    mode = hparams["mode"]
    
    # Extract paths and general config
    corpus_path = paths["corpus_path"]
    save_dir = paths["save_dir"]
    log_dir = paths["log_dir"]
    experiment_name = general["experiment_name"]
    create_sample = general["create_sample"] or args.create_sample
    
    # Create sample corpus if requested and doesn't exist
    if create_sample and not os.path.exists(corpus_path):
        print(f"Creating sample corpus at {corpus_path}")
        create_sample_corpus(corpus_path)
    
    # Check if corpus exists
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file {corpus_path} not found.")
        print("Use --create_sample to create a sample corpus or provide a valid corpus file.")
        return
    
    # Load and tokenize text
    print("Loading and tokenizing text...")
    text = load_text_corpus(corpus_path)
    
    # Create tokenizer and build vocabulary
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_type = tokenizer_config.get("type", "word")
    
    if tokenizer_type == "word":
        tokenizer = TokenizerFactory.create_tokenizer("word", vocab_size=vocab_size)
    elif tokenizer_type == "bpe":
        bpe_options = tokenizer_config.get("bpe_options", {})
        tokenizer = TokenizerFactory.create_tokenizer("bpe", vocab_size=vocab_size, **bpe_options)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    print(f"Using {tokenizer_type} tokenizer...")
    tokenizer.build_vocab(text)
    
    # Save vocabulary
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_vocab(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Encode text
    token_ids = tokenizer.encode(text)
    print(f"Encoded text length: {len(token_ids)} tokens")
    
    # Create data module
    data_module = TextDataModule(
        token_ids=token_ids,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers
    )
    
    # Create model
    model = TransformerLightningModule(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps
    )
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing (matching notebook format)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{experiment_name}-epoch={{epoch:02d}}-val_loss={{val/loss:.3f}}",
        monitor=monitor,
        mode=mode,
        auto_insert_metric_name=False,  # Prevents the name 'val/loss=' from being prepended
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=None
    )
    
    # Create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print model summary (matching notebook format)
    print("\nModel Summary:")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check data sizes (matching notebook)
    print("Data sizes:")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Train split: {train_split}")
    print(f"Val split: {val_split}")
    
    # Calculate split sizes
    total_len = len(token_ids)
    train_end = int(total_len * train_split)
    val_end = int(total_len * (train_split + val_split))
    
    print(f"Train tokens: {train_end}")
    print(f"Val tokens: {val_end - train_end}")
    print(f"Test tokens: {total_len - val_end}")
    
    # Check if validation data is sufficient
    val_tokens = val_end - train_end
    if val_tokens < sequence_length:
        print(f"WARNING: Validation data has only {val_tokens} tokens, less than sequence length {sequence_length}")
        print("This will cause validation to fail. Consider using a larger corpus or adjusting splits.")
    
    # Train model
    print("\nStarting training...")
    print("With the updated parameters:")
    print(f"- Sequence length: {sequence_length}")
    print(f"- Train split: {train_split}")
    print(f"- Val split: {val_split}")
    print(f"- Monitor: {monitor}")
    print()
    
    trainer.fit(model, data_module)
    
    # Test model
    print("\nTesting model...")
    trainer.test(model, data_module)
    
    # Save best model (matching notebook behavior)
    if checkpoint_callback.best_model_path:
        # Copy the best model to a final location with version number
        version = trainer.logger.version
        final_model_path = os.path.join(save_dir, f"{experiment_name}-best-v{version:02d}.ckpt")
        shutil.copy2(checkpoint_callback.best_model_path, final_model_path)
        print(f"Best model copied to {final_model_path}")
    else:
        print("No best model found, saving current model as final")
        version = trainer.logger.version
        final_model_path = os.path.join(save_dir, f"{experiment_name}-final-v{version:02d}.ckpt")
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    print("\nTraining completed!")
    print(f"Checkpoints saved in: {save_dir}")
    print(f"Logs saved in: {log_dir}")
    print(f"Vocabulary saved in: {vocab_path}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best score: {checkpoint_callback.best_model_score}")
    print(f"Final model saved as: {final_model_path}")
    
    # Print instructions for running the app
    print("\n" + "="*50)
    print("To run the Gradio app with your trained model:")
    print(f"python -m src.app.gradio_app --model_path {final_model_path} --vocab_path {vocab_path}")
    print("="*50)


if __name__ == "__main__":
    main()
