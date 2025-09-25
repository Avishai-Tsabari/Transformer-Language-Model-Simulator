"""
PyTorch Lightning module for transformer training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Perplexity
from typing import Any, Dict, Optional, Tuple
import math

from .transformer import TransformerLM


class TransformerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for transformer language model training.
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 max_len: int = 5000,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000):
        """
        Initialize the lightning module.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Metrics
        self.train_perplexity = Perplexity(ignore_index=0)
        self.val_perplexity = Perplexity(ignore_index=0)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, mask)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        input_ids, target_ids = batch
        
        # Forward pass
        logits = self(input_ids)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Calculate perplexity
        perplexity = self.train_perplexity(logits, target_ids)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        input_ids, target_ids = batch
        
        # Forward pass
        logits = self(input_ids)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Calculate perplexity
        perplexity = self.val_perplexity(logits, target_ids)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            batch_idx: Batch index
            
        Returns:
            Test loss
        """
        input_ids, target_ids = batch
        
        # Forward pass
        logits = self(input_ids)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                return 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / 
                                          (self.trainer.max_steps - self.warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset perplexity metric
        self.train_perplexity.reset()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Reset perplexity metric
        self.val_perplexity.reset()
    
    def generate_text(self, 
                     prompt: str,
                     tokenizer,
                     max_length: int = 100,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            tokenizer: Tokenizer instance
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Generated text
        """
        self.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text
    
    def get_next_token_probabilities(self,
                                   input_ids: torch.Tensor,
                                   temperature: float = 1.0,
                                   top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get probabilities for next token prediction.
        
        Args:
            input_ids: Input token indices
            temperature: Sampling temperature
            top_k: Top-k filtering
            
        Returns:
            Tuple of (top_k_tokens, top_k_probabilities)
        """
        self.eval()
        
        with torch.no_grad():
            # Get logits for next token
            logits = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            else:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, 10)
            
            # Convert to probabilities
            probabilities = F.softmax(top_k_logits, dim=-1)
            
        return top_k_indices[0], probabilities[0]


if __name__ == "__main__":
    # Test lightning module
    vocab_size = 1000
    model = TransformerLightningModule(vocab_size=vocab_size)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test training step
    loss = model.training_step((input_ids, target_ids), 0)
    print(f"Training loss: {loss}")
    
    # Test validation step
    val_loss = model.validation_step((input_ids, target_ids), 0)
    print(f"Validation loss: {val_loss}")
    
    print("Lightning module test completed successfully!")
