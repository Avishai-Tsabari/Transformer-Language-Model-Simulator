"""
Tokenizer and vocabulary builder for text corpus.
"""
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pickle
import os


class SimpleTokenizer:
    """
    A simple word-level tokenizer with vocabulary building capabilities.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_built = False
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text corpus.
        
        Args:
            text: Input text corpus
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split text into words
        words = text.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get most frequent words
        most_common_words = word_counts.most_common(self.vocab_size - 4)  # Reserve space for special tokens
        
        # Build vocabulary
        self.word_to_idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        
        # Add most common words
        for i, (word, _) in enumerate(most_common_words):
            self.word_to_idx[word] = i + 4
            
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_idx)} tokens")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
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
            
        text = self._clean_text(text)
        words = text.split()
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.SOS_TOKEN])
            
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx[self.UNK_TOKEN])
                
        if add_special_tokens:
            tokens.append(self.word_to_idx[self.EOS_TOKEN])
            
        return tokens
    
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
            
        words = []
        for token_id in token_ids:
            if token_id in self.idx_to_word:
                word = self.idx_to_word[token_id]
                if skip_special_tokens and word in [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    continue
                words.append(word)
                
        return " ".join(words)
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = vocab_data['vocab_size']
        self.vocab_built = True
        print(f"Vocabulary loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token indices."""
        return {
            'PAD': self.word_to_idx[self.PAD_TOKEN],
            'UNK': self.word_to_idx[self.UNK_TOKEN],
            'SOS': self.word_to_idx[self.SOS_TOKEN],
            'EOS': self.word_to_idx[self.EOS_TOKEN]
        }


def load_text_corpus(filepath: str) -> str:
    """
    Load text corpus from file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        Text content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_sample_corpus(filepath: str = "sample_corpus.txt") -> None:
    """
    Create a sample corpus for training.
    
    Args:
        filepath: Path to save the sample corpus
    """
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training a small language model.
    Machine learning is a fascinating field that combines mathematics, statistics, and computer science.
    Natural language processing allows computers to understand and generate human language.
    Transformers have revolutionized the field of deep learning with their attention mechanisms.
    PyTorch is a popular deep learning framework that provides flexibility and ease of use.
    Python is a versatile programming language used in data science and machine learning.
    Deep learning models require large amounts of data and computational resources.
    The attention mechanism allows models to focus on relevant parts of the input sequence.
    Positional encoding helps transformers understand the order of tokens in a sequence.
    Training neural networks involves forward propagation, loss computation, and backpropagation.
    Gradient descent is an optimization algorithm used to minimize the loss function.
    Regularization techniques help prevent overfitting in machine learning models.
    Cross-validation is a technique used to evaluate model performance on unseen data.
    Feature engineering involves selecting and transforming input variables for better model performance.
    Hyperparameter tuning is the process of finding the best configuration for a model.
    """
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"Sample corpus created at {filepath}")


if __name__ == "__main__":
    # Create sample corpus
    create_sample_corpus()
    
    # Test tokenizer
    text = load_text_corpus("sample_corpus.txt")
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(text)
    
    # Test encoding/decoding
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
