"""
Dataset utilities for text data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random
import pytorch_lightning as pl


class TextDataset(Dataset):
    """
    Dataset for text data with sliding window approach.
    """
    
    def __init__(self, 
                 token_ids: List[int], 
                 sequence_length: int = 64,
                 stride: int = 1):
        """
        Initialize dataset.
        
        Args:
            token_ids: List of token indices
            sequence_length: Length of input sequences
            stride: Step size for sliding window
        """
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Create sequence indices
        self.sequences = []
        for i in range(0, len(token_ids) - sequence_length, stride):
            self.sequences.append((i, i + sequence_length))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target (shifted by 1).
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        start, end = self.sequences[idx]
        sequence = self.token_ids[start:end]
        
        # Input is sequence[:-1], target is sequence[1:]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_seq, target_seq


def create_data_loaders(token_ids: List[int],
                       sequence_length: int = 64,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        token_ids: List of token indices
        sequence_length: Length of input sequences
        batch_size: Batch size
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split indices
    total_len = len(token_ids)
    train_end = int(total_len * train_split)
    val_end = int(total_len * (train_split + val_split))
    
    # Split data
    train_tokens = token_ids[:train_end]
    val_tokens = token_ids[train_end:val_end]
    test_tokens = token_ids[val_end:]
    
    # Create datasets
    train_dataset = TextDataset(train_tokens, sequence_length)
    val_dataset = TextDataset(val_tokens, sequence_length)
    test_dataset = TextDataset(test_tokens, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching sequences.
    
    Args:
        batch: List of (input_seq, target_seq) tuples
        
    Returns:
        Batched input and target tensors
    """
    input_seqs, target_seqs = zip(*batch)
    
    # Pad sequences to the same length
    input_tensor = torch.nn.utils.rnn.pad_sequence(
        input_seqs, 
        batch_first=True, 
        padding_value=0
    )
    target_tensor = torch.nn.utils.rnn.pad_sequence(
        target_seqs, 
        batch_first=True, 
        padding_value=0
    )
    
    return input_tensor, target_tensor


class TextDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for text data.
    """
    
    def __init__(self,
                 token_ids: List[int],
                 sequence_length: int = 64,
                 batch_size: int = 32,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 num_workers: int = 0):
        """
        Initialize data module.
        
        Args:
            token_ids: List of token indices
            sequence_length: Length of input sequences
            batch_size: Batch size
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of worker processes
        """
        super().__init__()
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data loaders."""
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            self.token_ids,
            self.sequence_length,
            self.batch_size,
            self.train_split,
            self.val_split,
            self.num_workers
        )
    
    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return self.train_loader
    
    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return self.val_loader
    
    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return self.test_loader


if __name__ == "__main__":
    # Test dataset creation
    from tokenizer import SimpleTokenizer, load_text_corpus
    
    # Load sample data
    text = load_text_corpus("sample_corpus.txt")
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(text)
    
    # Encode text
    token_ids = tokenizer.encode(text)
    
    # Create dataset
    dataset = TextDataset(token_ids, sequence_length=32)
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    input_seq, target_seq = dataset[0]
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    print(f"Input: {input_seq[:10]}")
    print(f"Target: {target_seq[:10]}")
