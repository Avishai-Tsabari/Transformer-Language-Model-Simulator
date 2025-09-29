"""
Simple BPE tokenizer using the tokenizers library for better performance and reliability.
"""
import pickle
import os
from typing import List, Dict, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class SimpleBPETokenizer:
    """
    Simple BPE tokenizer using the tokenizers library.
    """
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 min_frequency: int = 2,
                 max_merges: Optional[int] = None,
                 special_tokens: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a merge
            max_merges: Maximum number of merges (None = no limit, ignored in this implementation)
            special_tokens: List of special tokens
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_merges = max_merges
        
        # Special tokens
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.special_tokens = special_tokens
        
        # Token IDs for easy access
        self.unk_token_id = None  # Will be set after vocab is built
        
        # Initialize tokenizer
        self.tokenizer = None
        self.vocab_built = False
        
    def build_vocab(self, text: str) -> None:
        """
        Build BPE vocabulary from text corpus.
        
        Args:
            text: Input text corpus
        """
        print("Building BPE vocabulary...")
        
        # Clean text
        text = self._clean_text(text)
        
        # Initialize BPE model
        model = BPE(unk_token="<UNK>")
        
        # Initialize trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(model)
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Train the tokenizer
        print("Training BPE tokenizer...")
        self.tokenizer.train_from_iterator([text], trainer)
        
        # Set up post-processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="<SOS> $A <EOS>",
            special_tokens=[
                ("<SOS>", self.tokenizer.get_vocab()["<SOS>"]),
                ("<EOS>", self.tokenizer.get_vocab()["<EOS>"])
            ]
        )
        
        # Set UNK token ID
        vocab = self.tokenizer.get_vocab()
        self.unk_token_id = vocab.get('<UNK>', None)
        
        self.vocab_built = True
        print(f"BPE vocabulary built with {self.tokenizer.get_vocab_size()} tokens")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text.strip()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token indices
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Clean text
        text = self._clean_text(text)
        
        # Encode
        encoding = self.tokenizer.encode(text)
        
        if add_special_tokens:
            return encoding.ids
        else:
            # Remove SOS and EOS tokens
            ids = encoding.ids
            if ids and ids[0] == self.tokenizer.get_vocab()["<SOS>"]:
                ids = ids[1:]
            if ids and ids[-1] == self.tokenizer.get_vocab()["<EOS>"]:
                ids = ids[:-1]
            return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text.
        
        Args:
            token_ids: List of token indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Decode
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
        # Clean up spacing around punctuation
        import re
        # Remove spaces before punctuation
        decoded = re.sub(r'\s+([.,!?;:])', r'\1', decoded)
        # Remove spaces after opening punctuation
        decoded = re.sub(r'([\'"\(])\s+', r'\1', decoded)
        # Remove spaces before closing punctuation
        decoded = re.sub(r'\s+([\'"\)])', r'\1', decoded)
        # Clean up multiple spaces
        decoded = re.sub(r'\s+', ' ', decoded).strip()
        
        return decoded
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Save tokenizer
        tokenizer_path = filepath.replace('.pkl', '_tokenizer.json')
        self.tokenizer.save(tokenizer_path)
        
        # Save additional metadata
        vocab_data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'max_merges': self.max_merges,
            'special_tokens': self.special_tokens,
            'tokenizer_path': tokenizer_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"BPE vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        # Load metadata
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.min_frequency = vocab_data['min_frequency']
        self.max_merges = vocab_data['max_merges']
        self.special_tokens = vocab_data['special_tokens']
        
        # Load tokenizer
        tokenizer_path = vocab_data['tokenizer_path']
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.vocab_built = True
        print(f"BPE vocabulary loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.vocab_built:
            return 0
        return self.tokenizer.get_vocab_size()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token indices."""
        if not self.vocab_built:
            return {}
        vocab = self.tokenizer.get_vocab()
        return {token: vocab[token] for token in self.special_tokens if token in vocab}
    
    def get_stats(self) -> Dict[str, any]:
        """Get tokenizer statistics."""
        if not self.vocab_built:
            return {}
        
        return {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'special_tokens': self.special_tokens,
            'min_frequency': self.min_frequency,
            'max_merges': self.max_merges
        }


if __name__ == "__main__":
    # Test BPE tokenizer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.tokenizer import load_text_corpus
    
    # Load sample text
    text = load_text_corpus("sample_corpus.txt")
    
    # Test with different configurations
    configs = [
        {"vocab_size": 100, "min_frequency": 2},
        {"vocab_size": 500, "min_frequency": 2},
        {"vocab_size": 1000, "min_frequency": 2},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing BPE with config: {config}")
        print(f"{'='*60}")
        
        tokenizer = SimpleBPETokenizer(**config)
        tokenizer.build_vocab(text)
        
        # Test encoding/decoding
        test_text = "hello world this is a test"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original: '{test_text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Stats: {tokenizer.get_stats()}")
