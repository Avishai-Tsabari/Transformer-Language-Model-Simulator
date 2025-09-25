"""
Training script for the transformer language model.
"""
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from src.data.tokenizer import SimpleTokenizer, load_text_corpus, create_sample_corpus
from src.data.dataset import TextDataModule
from src.model.lightning_module import TransformerLightningModule


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--corpus_path", type=str, default="sample_corpus.txt", 
                       help="Path to text corpus file")
    parser.add_argument("--vocab_size", type=int, default=1000, 
                       help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, 
                       help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, 
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, 
                       help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=1024, 
                       help="Feed-forward dimension")
    parser.add_argument("--sequence_length", type=int, default=64, 
                       help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, 
                       help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, 
                       help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001, 
                       help="Minimum change for early stopping")
    parser.add_argument("--warmup_steps", type=int, default=1000, 
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                       help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, 
                       help="Dropout rate")
    parser.add_argument("--train_split", type=float, default=0.8, 
                       help="Training data split")
    parser.add_argument("--val_split", type=float, default=0.1, 
                       help="Validation data split")
    parser.add_argument("--num_workers", type=int, default=0, 
                       help="Number of data loader workers")
    parser.add_argument("--accelerator", type=str, default="auto", 
                       help="Accelerator type (auto, cpu, gpu, etc.)")
    parser.add_argument("--devices", type=int, default=1, 
                       help="Number of devices")
    parser.add_argument("--precision", type=str, default="32", 
                       help="Precision (16, 32, bf16)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, 
                       help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, 
                       help="Gradient accumulation steps")
    parser.add_argument("--log_every_n_steps", type=int, default=50, 
                       help="Log every n steps")
    parser.add_argument("--val_check_interval", type=float, default=1.0, 
                       help="Validation check interval")
    parser.add_argument("--save_top_k", type=int, default=3, 
                       help="Number of best models to save")
    parser.add_argument("--monitor", type=str, default="val/loss", 
                       help="Metric to monitor for checkpointing")
    parser.add_argument("--mode", type=str, default="min", 
                       help="Mode for checkpointing (min or max)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                       help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                       help="Directory to save logs")
    parser.add_argument("--experiment_name", type=str, default="transformer_lm", 
                       help="Experiment name for logging")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample corpus if it doesn't exist")
    
    args = parser.parse_args()
    
    # Create sample corpus if requested and doesn't exist
    if args.create_sample and not os.path.exists(args.corpus_path):
        print(f"Creating sample corpus at {args.corpus_path}")
        create_sample_corpus(args.corpus_path)
    
    # Check if corpus exists
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus file {args.corpus_path} not found.")
        print("Use --create_sample to create a sample corpus or provide a valid corpus file.")
        return
    
    # Load and tokenize text
    print("Loading and tokenizing text...")
    text = load_text_corpus(args.corpus_path)
    
    # Create tokenizer and build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(text)
    
    # Save vocabulary
    vocab_path = os.path.join(args.save_dir, "vocab.pkl")
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save_vocab(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Encode text
    token_ids = tokenizer.encode(text)
    print(f"Encoded text length: {len(token_ids)} tokens")
    
    # Create data module
    data_module = TextDataModule(
        token_ids=token_ids,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    # Create model
    model = TransformerLightningModule(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        patience=args.patience,
        min_delta=args.min_delta,
        mode=args.mode,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"{args.experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor=args.monitor,
        mode=args.mode,
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=None
    )
    
    # Create trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test model
    print("\nTesting model...")
    trainer.test(model, data_module)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"{args.experiment_name}-final.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("\nTraining completed!")
    print(f"Checkpoints saved in: {args.save_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print(f"Vocabulary saved in: {vocab_path}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best score: {checkpoint_callback.best_model_score}")
    
    # Print instructions for running the app
    print("\n" + "="*50)
    print("To run the Gradio app with your trained model:")
    print(f"python -m src.app.gradio_app --model_path {checkpoint_callback.best_model_path} --vocab_path {vocab_path}")
    print("="*50)


if __name__ == "__main__":
    main()
